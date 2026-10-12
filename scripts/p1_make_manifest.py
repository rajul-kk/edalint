"""Build manifests/MANIFEST.json for the P1 data acquisition.

Re-runnable. Reads the downloaded data under E:/edalint_data (outside the repo), hashes what is present and records what is
not. For HuggingFace files the expected sha256 is the LFS oid reported by the HF API; the local hash is computed here and
compared. Hash results are cached in E:/edalint_data/_tools/sha_cache.json (keyed by path, size, mtime).

Usage:  python -I scripts/p1_make_manifest.py
"""
import hashlib, json, os, sys, time

ROOT = "E:/edalint_data"
TOOLS = ROOT + "/_tools"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manifests", "MANIFEST.json")
HF_N14_REV = "df74f33e302f927bf0548dc0d7936ca43b634f2a"
HF_N45_REV = "5683d95097bb27bb380f135101d7da6991025557"
GH_CN_COMMIT = "41ade1d7e913a418b4495b6b13630b26b0ff7279"
ICCAD_COMMIT = "bda90cbe53f973f28d7c96951af665f678a84180"

cache_path = TOOLS + "/sha_cache.json"
cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}


def sha256_file(p):
    st = os.stat(p)
    key = "%s|%d|%d" % (p, st.st_size, int(st.st_mtime))
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    cache[key] = h.hexdigest()
    json.dump(cache, open(cache_path, "w"))
    return cache[key]


def main():
    m = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "datasets": {}}

    # 1. ICCAD 2023 Problem C: git clone, every file hashed by E:/edalint_data/_tools/manifest.py
    files = json.load(open(TOOLS + "/iccad23_files.json"))
    m["datasets"]["iccad23_problem_c"] = {
        "url": "https://github.com/ASU-VDA-Lab/ML-for-IR-drop", "commit": ICCAD_COMMIT, "license": "BSD-3-Clause",
        "local_root": "E:/edalint_data/iccad23/repo", "n_files": len(files),
        "total_bytes": sum(v["size"] for v in files.values()),
        "obtained": True, "files": files}

    # 2. CircuitNet-N14: IR_drop_features + routability_features subsets of the HF repo
    lfs = json.load(open(TOOLS + "/cn_hf_lfs.json"))
    n14 = {}
    for k in sorted(lfs):
        if not (k.startswith("CircuitNet-N14/IR_drop_features/") or k.startswith("CircuitNet-N14/routability_features/")):
            continue
        p = ROOT + "/circuitnet_n14/" + k
        e = {"url": "https://huggingface.co/datasets/CircuitNet/CircuitNet/resolve/%s/%s" % (HF_N14_REV, k),
             "size": lfs[k]["size"], "sha256_expected_hf_lfs": lfs[k]["sha256"]}
        if os.path.exists(p) and os.path.getsize(p) == lfs[k]["size"]:
            e["sha256"] = sha256_file(p)
            e["sha256_matches_hf"] = e["sha256"] == lfs[k]["sha256"]
            e["status"] = "downloaded"
        else:
            e["status"] = "not yet downloaded"
        n14[k] = e
    m["datasets"]["circuitnet_n14"] = {
        "repo": "CircuitNet/CircuitNet (HF dataset)", "revision": HF_N14_REV, "license": "BSD-3-Clause",
        "local_root": "E:/edalint_data/circuitnet_n14", "subsets": ["IR_drop_features", "routability_features"],
        "n_files": len(n14), "total_bytes_expected": sum(v["size"] for v in n14.values()),
        "n_downloaded": sum(v["status"] == "downloaded" for v in n14.values()),
        "files": n14}

    # 3. CircuitNet-N28: not obtained
    m["datasets"]["circuitnet_n28"] = {
        "obtained": False,
        "reason": "Google Drive public folder lists (gdown) but every file download returns 'Quota exceeded' "
                  "(HTTP 200 html page, 2026-10-12). Baidu Netdisk needs an account. No access control was bypassed.",
        "drive_folder": "https://drive.google.com/drive/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB",
        "baidu": "https://pan.baidu.com/s/1udXVZnfjqniH9paKfyc2eQ?pwd=ijdh", "sizes_known": False}

    # 4. CircuitNet 3.0 (N45): README only; zip not downloaded (no map tasks)
    rp = ROOT + "/circuitnet3/README.md"
    m["datasets"]["circuitnet_3_0"] = {
        "repo": "SKLP-EDA-LAB/CircuitNet3.0 (HF dataset)", "revision": HF_N45_REV, "license": "apache-2.0",
        "obtained": False, "downloaded": {"README.md": {"size": os.path.getsize(rp), "sha256": sha256_file(rp)}},
        "not_downloaded": {"circuitNetv3.zip": {"size": 1032704519}},
        "reason": "README and zip central directory (read by HTTP range, 57,975 entries) show RTL/netlist/timing/power only; "
                  "no IR-drop, congestion or DRC map task."}

    # 5. Original-loader code and documentation snapshots used for the format notes
    docs = {}
    dc = ROOT + "/docs_cache"
    for dp, dn, fn in os.walk(dc):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, dc).replace(os.sep, "/")
            if f.endswith(".html"):
                continue  # text copies are kept; raw html omitted from the manifest
            docs[rel] = {"size": os.path.getsize(p), "sha256": sha256_file(p)}
    m["datasets"]["docs_snapshots"] = {
        "note": "GitHub circuitnet/CircuitNet @ %s (raw files), circuitnet.github.io pages saved as text, GitHub issues JSON "
                "(fetched 2026-10-12)" % GH_CN_COMMIT, "local_root": dc, "files": docs}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(m, open(OUT, "w"), indent=1)
    d = m["datasets"]["circuitnet_n14"]
    print("N14: %d/%d files downloaded" % (d["n_downloaded"], d["n_files"]))


if __name__ == "__main__":
    sys.exit(main())
