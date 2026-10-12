"""File-name-level inventory of the downloaded CircuitNet-N14 tarballs (no array is loaded, no statistic is computed).

For each downloaded IR_drop_features / routability_features tarball: number of files per feature directory, and the coverage
of the label directory (IR_drop, DRC_all) by the official split CSVs (train_N14.csv / test_N14.csv, GitHub
circuitnet/CircuitNet @ 41ade1d, copies in E:/edalint_data/docs_cache/cn_gh/...). Writes manifests/n14_inventory.json.
Re-runnable; tarballs already in the output are skipped. Usage: python -I scripts/p1_n14_inventory.py
"""
import collections, gzip, io, json, os, sys, tarfile

ROOT = "E:/edalint_data/circuitnet_n14/CircuitNet-N14"
CSV = "E:/edalint_data/docs_cache/cn_gh/routability_ir_drop_prediction/files/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "manifests", "n14_inventory.json")


def stems(path):
    return [l.split(",")[0].split("/")[1][:-4] for l in open(path).read().split()]


def main():
    tr, te = set(stems(CSV + "train_N14.csv")), set(stems(CSV + "test_N14.csv"))
    inv = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for sub in ("IR_drop_features", "routability_features"):
        d = os.path.join(ROOT, sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not f.endswith(".tar.gz"):
                continue
            key = sub + "/" + f
            if key in inv:
                continue
            p = os.path.join(d, f)
            try:
                if os.path.getsize(p) != json.load(open("E:/edalint_data/_tools/cn_hf_lfs.json"))["CircuitNet-N14/" + key]["size"]:
                    continue  # incomplete
            except KeyError:
                continue
            counts, names = collections.Counter(), collections.defaultdict(set)
            # tarfile's own 'r|gz' is slow (8 MB/s); a large-buffer GzipFile gives about 130 MB/s (see datasets/circuitnet.py)
            raw = open(p, 'rb')
            gz = gzip.GzipFile(fileobj=raw)
            tf = tarfile.open(fileobj=io.BufferedReader(gz, buffer_size=1 << 24), mode='r|', bufsize=1 << 24)
            for m in tf:
                if m.isfile():
                    dd = "/".join(m.name.split("/")[:-1])
                    counts[dd] += 1
                    names[dd].add(os.path.basename(m.name)[:-4])
            tf.close(); gz.close(); raw.close()
            design = f[:-7]
            lab = [x for x in names if x.endswith("/IR_drop") or x.endswith("/DRC/DRC_all")][0]
            label = names[lab]
            allf = set.union(*[v for k, v in names.items() if "DRC_seperated" not in k])
            inv[key] = {
                "design": design, "files_per_dir": {k.lstrip("./"): v for k, v in counts.items() if "DRC_seperated" not in k},
                "n_DRC_seperated_types": sum(1 for k in counts if "DRC_seperated" in k),
                "label_dir": lab.lstrip("./"), "n_label": len(label), "n_union_of_feature_dirs": len(allf),
                "n_train_csv": len(label & tr), "n_test_csv": len(label & te), "n_label_in_neither_csv": len(label - tr - te),
                "n_csv_names_not_in_label_dir": len({x for x in tr | te if x.split("_freq")[0] == design} - label)}
            json.dump(inv, open(OUT, "w"), indent=1)
            print(key, inv[key]["n_label"], "train", inv[key]["n_train_csv"], "test", inv[key]["n_test_csv"])


if __name__ == "__main__":
    sys.exit(main())
