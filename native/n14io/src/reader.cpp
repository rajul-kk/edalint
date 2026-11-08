// edalint_n14io._reader: stream a CircuitNet-N14 .tar.gz and return the numpy arrays of its .npz members.
//
// Same semantics as edalint.datasets.circuitnet.stream_tarball: one sequential pass; members "<design>/<dir>/<name>.npz"
// whose <dir> is in dir_to_feature are reported (presence); those whose <name> is wanted (or all, if wanted is None) also
// carry the array of their "data" entry. Everything that is not Python (inflate, tar and zip parsing, the inner inflate of
// compressed npz entries) runs with the GIL released, so several tarballs can be read from Python threads at once.
//
// Formats: gzip (possibly multi-member) via zlib; ustar/GNU tar with GNU long names ('L') and pax path records ('x');
// npz = zip with stored or deflated entries, read through the central directory (zip64 sizes and offsets supported);
// npy format 1.0/2.0/3.0 with a little-endian or byte-order-free dtype, C or Fortran order.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------------------------- gzip byte source
class GzipSource {
  public:
    explicit GzipSource(const std::string& path) : in_(1 << 22) {
        f_ = std::fopen(path.c_str(), "rb");
        if (!f_) throw std::runtime_error("cannot open " + path);
        std::memset(&z_, 0, sizeof(z_));
        if (inflateInit2(&z_, 16 + MAX_WBITS) != Z_OK) throw std::runtime_error("inflateInit2 failed");
    }
    ~GzipSource() {
        inflateEnd(&z_);
        if (f_) std::fclose(f_);
    }
    GzipSource(const GzipSource&) = delete;
    GzipSource& operator=(const GzipSource&) = delete;

    // Read exactly n bytes into out (or discard them if out is null). Returns false on a clean end of data before any
    // byte; throws on truncation mid-request or corrupt data.
    bool read(uint8_t* out, size_t n) {
        size_t got = 0;
        while (got < n) {
            if (z_.avail_in == 0 && !refill()) {
                if (got == 0 && finished_) return false;
                throw std::runtime_error("truncated gzip stream");
            }
            uint8_t sink[1 << 16];
            uint8_t* dst = out ? out + got : sink;
            size_t want = out ? n - got : std::min(n - got, sizeof(sink));
            z_.next_out = dst;
            z_.avail_out = static_cast<uInt>(std::min<size_t>(want, 1u << 30));
            int r = inflate(&z_, Z_NO_FLUSH);
            got += (z_.next_out - dst);
            if (r == Z_STREAM_END) {
                // concatenated gzip members: start the next one if more input follows
                if (z_.avail_in == 0 && !refill()) {
                    finished_ = true;
                    if (got < n) {
                        if (got == 0) return false;
                        throw std::runtime_error("truncated gzip stream");
                    }
                    return true;
                }
                if (inflateReset(&z_) != Z_OK) throw std::runtime_error("inflateReset failed");
            } else if (r == Z_BUF_ERROR) {
                if (z_.avail_in == 0) continue;
            } else if (r != Z_OK) {
                throw std::runtime_error(std::string("corrupt gzip data: ") + (z_.msg ? z_.msg : "zlib error"));
            }
        }
        return true;
    }
    bool skip(size_t n) { return read(nullptr, n); }

  private:
    bool refill() {
        if (eof_) return false;
        size_t k = std::fread(in_.data(), 1, in_.size(), f_);
        if (k == 0) {
            eof_ = true;
            return false;
        }
        z_.next_in = in_.data();
        z_.avail_in = static_cast<uInt>(k);
        return true;
    }
    std::FILE* f_ = nullptr;
    std::vector<uint8_t> in_;
    z_stream z_;
    bool eof_ = false, finished_ = false;
};

// ---------------------------------------------------------------------------------------------- tar
uint64_t tar_number(const uint8_t* p, size_t n) {
    if (p[0] & 0x80) {  // GNU base-256 for large sizes
        uint64_t v = p[0] & 0x7f;
        for (size_t i = 1; i < n; ++i) v = (v << 8) | p[i];
        return v;
    }
    uint64_t v = 0;
    size_t i = 0;
    while (i < n && (p[i] == ' ' || p[i] == 0)) ++i;
    for (; i < n && p[i] >= '0' && p[i] <= '7'; ++i) v = v * 8 + (p[i] - '0');
    return v;
}

std::string cstr(const uint8_t* p, size_t n) {
    size_t k = 0;
    while (k < n && p[k]) ++k;
    return std::string(reinterpret_cast<const char*>(p), k);
}

std::optional<std::string> pax_path(const std::string& rec) {
    // records "<len> key=value\n"
    size_t i = 0;
    std::optional<std::string> path;
    while (i < rec.size()) {
        size_t sp = rec.find(' ', i);
        if (sp == std::string::npos) break;
        size_t len = std::stoul(rec.substr(i, sp - i));
        if (len == 0 || i + len > rec.size()) break;
        std::string kv = rec.substr(sp + 1, len - (sp - i) - 2);  // drop the trailing newline
        if (kv.compare(0, 5, "path=") == 0) path = kv.substr(5);
        i += len;
    }
    return path;
}

// ---------------------------------------------------------------------------------------------- npz / npy
uint16_t le16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t le32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
uint64_t le64(const uint8_t* p) { return uint64_t(le32(p)) | (uint64_t(le32(p + 4)) << 32); }

// The bytes of entry `want` of a zip archive held in memory (stored or deflated).
std::vector<uint8_t> zip_entry(const std::vector<uint8_t>& z, const std::string& want) {
    if (z.size() < 22) throw std::runtime_error("npz too small");
    size_t eocd = std::string::npos;
    for (size_t i = z.size() - 22 + 1; i-- > (z.size() > 65557 ? z.size() - 65557 : 0);) {
        if (le32(&z[i]) == 0x06054b50u) {
            eocd = i;
            break;
        }
    }
    if (eocd == std::string::npos) throw std::runtime_error("npz: no end of central directory");
    uint64_t cd_off = le32(&z[eocd + 16]), n_entries = le16(&z[eocd + 10]);
    if ((cd_off == 0xffffffffu || n_entries == 0xffff) && eocd >= 20 && le32(&z[eocd - 20]) == 0x07064b50u) {
        uint64_t z64 = le64(&z[eocd - 20 + 8]);
        if (z64 + 56 > z.size() || le32(&z[z64]) != 0x06064b50u) throw std::runtime_error("npz: bad zip64 record");
        n_entries = le64(&z[z64 + 32]);
        cd_off = le64(&z[z64 + 48]);
    }
    size_t p = cd_off;
    for (uint64_t e = 0; e < n_entries; ++e) {
        if (p + 46 > z.size() || le32(&z[p]) != 0x02014b50u) throw std::runtime_error("npz: bad central directory");
        uint16_t method = le16(&z[p + 10]);
        uint64_t csize = le32(&z[p + 20]), usize = le32(&z[p + 24]), loff = le32(&z[p + 42]);
        uint16_t nlen = le16(&z[p + 28]), xlen = le16(&z[p + 30]), clen = le16(&z[p + 32]);
        std::string name(reinterpret_cast<const char*>(&z[p + 46]), nlen);
        // zip64 extra field: the 0xffffffff fields follow in order usize, csize, offset
        size_t x = p + 46 + nlen, xend = x + xlen;
        while (x + 4 <= xend) {
            uint16_t id = le16(&z[x]), sz = le16(&z[x + 2]);
            if (id == 0x0001) {
                size_t q = x + 4;
                if (usize == 0xffffffffu) { usize = le64(&z[q]); q += 8; }
                if (csize == 0xffffffffu) { csize = le64(&z[q]); q += 8; }
                if (loff == 0xffffffffu) { loff = le64(&z[q]); }
            }
            x += 4 + sz;
        }
        if (name == want) {
            if (loff + 30 > z.size() || le32(&z[loff]) != 0x04034b50u) throw std::runtime_error("npz: bad local header");
            size_t data = loff + 30 + le16(&z[loff + 26]) + le16(&z[loff + 28]);
            if (data + csize > z.size()) throw std::runtime_error("npz: entry runs past the archive");
            std::vector<uint8_t> out(usize);
            if (method == 0) {
                if (csize != usize) throw std::runtime_error("npz: stored entry with csize != usize");
                std::memcpy(out.data(), &z[data], usize);
            } else if (method == 8) {
                z_stream s;
                std::memset(&s, 0, sizeof(s));
                if (inflateInit2(&s, -MAX_WBITS) != Z_OK) throw std::runtime_error("inflateInit2 failed");
                s.next_in = const_cast<uint8_t*>(&z[data]);
                s.avail_in = static_cast<uInt>(csize);
                s.next_out = out.data();
                s.avail_out = static_cast<uInt>(usize);
                int r = inflate(&s, Z_FINISH);
                inflateEnd(&s);
                if (r != Z_STREAM_END || s.total_out != usize) throw std::runtime_error("npz: bad deflate entry");
            } else {
                throw std::runtime_error("npz: unsupported compression method " + std::to_string(method));
            }
            return out;
        }
        p += 46 + nlen + xlen + clen;
    }
    throw std::runtime_error("npz has no entry " + want);
}

struct Npy {
    std::string descr;
    bool fortran = false;
    std::vector<py::ssize_t> shape;
    size_t offset = 0;
};

std::string dict_value(const std::string& h, const std::string& key) {
    size_t k = h.find("'" + key + "'");
    if (k == std::string::npos) throw std::runtime_error("npy header without " + key);
    size_t c = h.find(':', k);
    size_t s = h.find_first_not_of(" ", c + 1);
    if (h[s] == '\'') return h.substr(s + 1, h.find('\'', s + 1) - s - 1);
    if (h[s] == '(') return h.substr(s + 1, h.find(')', s) - s - 1);
    size_t e = h.find_first_of(",}", s);
    return h.substr(s, e - s);
}

Npy parse_npy(const std::vector<uint8_t>& b) {
    static const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (b.size() < 10 || std::memcmp(b.data(), magic, 6) != 0) throw std::runtime_error("not an npy array");
    Npy n;
    size_t hlen, hstart;
    if (b[6] == 1) {
        hlen = le16(&b[8]);
        hstart = 10;
    } else {
        hlen = le32(&b[8]);
        hstart = 12;
    }
    if (hstart + hlen > b.size()) throw std::runtime_error("npy header runs past the data");
    std::string h(reinterpret_cast<const char*>(&b[hstart]), hlen);
    n.descr = dict_value(h, "descr");
    n.fortran = dict_value(h, "fortran_order").find("True") != std::string::npos;
    std::string sh = dict_value(h, "shape");
    size_t i = 0;
    while (i < sh.size()) {
        size_t j = sh.find(',', i);
        std::string t = sh.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (t.find_first_of("0123456789") != std::string::npos) n.shape.push_back(std::stoll(t));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    if (!n.descr.empty() && n.descr[0] == '>') throw std::runtime_error("big-endian npy arrays are not supported");
    n.offset = hstart + hlen;
    return n;
}

// ---------------------------------------------------------------------------------------------- the reader
struct Member {
    std::string feature, name;
    bool has_data = false;
    std::vector<uint8_t> npy;  // the bytes of data.npy (when has_data)
};

class Reader {
  public:
    Reader(const std::string& path, std::string design, std::unordered_map<std::string, std::string> dir_to_feature,
           std::optional<std::unordered_set<std::string>> wanted)
        : src_(path), design_(std::move(design)), dirs_(std::move(dir_to_feature)), wanted_(std::move(wanted)) {}

    // Next matching member, or nullopt at the end of the archive. Runs without the GIL.
    std::optional<Member> next() {
        uint8_t hdr[512];
        std::string long_name;
        while (true) {
            if (!src_.read(hdr, 512)) return std::nullopt;
            bool zero = true;
            for (uint8_t c : hdr) zero &= (c == 0);
            if (zero) return std::nullopt;  // end-of-archive block
            const uint64_t size = tar_number(hdr + 124, 12);
            const uint64_t padded = (size + 511) / 512 * 512;
            const char type = static_cast<char>(hdr[156]);
            if (type == 'L' || type == 'x') {  // GNU long name / pax header: their data names the next entry
                std::string data(size, '\0');
                if (!src_.read(reinterpret_cast<uint8_t*>(data.data()), size) || !src_.skip(padded - size))
                    throw std::runtime_error("truncated tar header record");
                if (type == 'L') {
                    long_name = data.substr(0, data.find('\0'));
                } else if (auto p = pax_path(data)) {
                    long_name = *p;
                }
                continue;
            }
            if (type == 'g') {  // pax global header: no per-file meaning here
                if (!src_.skip(padded)) throw std::runtime_error("truncated tar");
                continue;
            }
            std::string name = long_name;
            long_name.clear();
            if (name.empty()) {
                name = cstr(hdr, 100);
                if (std::memcmp(hdr + 257, "ustar", 5) == 0) {
                    std::string prefix = cstr(hdr + 345, 155);
                    if (!prefix.empty()) name = prefix + "/" + name;
                }
            }
            const bool regular = type == '0' || type == '\0' || type == '7';
            auto m = regular ? match(name) : std::nullopt;
            if (!m) {
                if (!src_.skip(padded)) throw std::runtime_error("truncated tar member " + name);
                continue;
            }
            if (m->has_data) {
                std::vector<uint8_t> z(size);
                if (!src_.read(z.data(), size)) throw std::runtime_error("truncated tar member " + name);
                m->npy = zip_entry(z, "data.npy");
            } else if (!src_.skip(size)) {
                throw std::runtime_error("truncated tar member " + name);
            }
            if (!src_.skip(padded - size)) throw std::runtime_error("truncated tar member " + name);
            return m;
        }
    }

  private:
    std::optional<Member> match(const std::string& path) {
        if (path.size() < 4 || path.compare(path.size() - 4, 4, ".npz") != 0) return std::nullopt;
        size_t slash = path.rfind('/');
        std::string d = slash == std::string::npos ? "" : path.substr(0, slash);
        std::string fn = slash == std::string::npos ? path : path.substr(slash + 1);
        if (d.compare(0, 2, "./") == 0) d = d.substr(2);
        if (d.compare(0, design_.size() + 1, design_ + "/") == 0) d = d.substr(design_.size() + 1);
        auto it = dirs_.find(d);
        if (it == dirs_.end()) return std::nullopt;
        Member m;
        m.feature = it->second;
        m.name = fn.substr(0, fn.size() - 4);
        m.has_data = !wanted_ || wanted_->count(m.name) > 0;
        return m;
    }
    GzipSource src_;
    std::string design_;
    std::unordered_map<std::string, std::string> dirs_;
    std::optional<std::unordered_set<std::string>> wanted_;
};

py::array to_numpy(const std::vector<uint8_t>& b) {
    Npy n = parse_npy(b);
    py::dtype dt(n.descr);
    const auto item = static_cast<py::ssize_t>(dt.itemsize());
    py::ssize_t count = 1;
    for (auto s : n.shape) count *= s;
    if (n.offset + static_cast<size_t>(count * item) > b.size()) throw std::runtime_error("npy data is truncated");
    std::vector<py::ssize_t> strides(n.shape.size());
    py::ssize_t st = item;
    if (n.fortran) {
        for (size_t i = 0; i < n.shape.size(); ++i) { strides[i] = st; st *= n.shape[i]; }
    } else {
        for (size_t i = n.shape.size(); i-- > 0;) { strides[i] = st; st *= n.shape[i]; }
    }
    py::array a(dt, n.shape, strides);
    std::memcpy(a.mutable_data(), b.data() + n.offset, static_cast<size_t>(count * item));
    return a;
}

}  // namespace

PYBIND11_MODULE(_reader, m) {
    m.doc() = "Native CircuitNet-N14 tarball reader (see edalint_n14io)";
    py::class_<Reader>(m, "Reader")
        .def(py::init([](const std::string& path, const std::string& design,
                         std::unordered_map<std::string, std::string> dirs, std::optional<py::iterable> wanted) {
                 std::optional<std::unordered_set<std::string>> w;
                 if (wanted) {
                     w.emplace();
                     for (auto h : *wanted) w->insert(py::cast<std::string>(h));
                 }
                 return new Reader(path, design, std::move(dirs), std::move(w));
             }),
             py::arg("path"), py::arg("design"), py::arg("dir_to_feature"), py::arg("wanted") = py::none())
        .def("next", [](Reader& r) -> py::object {
            std::optional<Member> mem;
            {
                py::gil_scoped_release release;
                mem = r.next();
            }
            if (!mem) return py::none();
            py::object arr = mem->has_data ? py::object(to_numpy(mem->npy)) : py::object(py::none());
            return py::make_tuple(mem->feature, mem->name, arr);
        });
}
