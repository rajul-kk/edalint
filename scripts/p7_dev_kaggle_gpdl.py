"""Kaggle kernel entry (DEVELOPMENT, train-only, p7dev_kaggle): gpdl T4 benchmark and held-out check. Body: scripts/p7_dev_kaggle_common.py."""
import glob
import sys

hits = sorted(glob.glob('/kaggle/input/**/scripts/p7_dev_kaggle_common.py', recursive=True))
if not hits:
    raise SystemExit('code dataset not attached')
sys.path.insert(0, hits[0].rsplit('/', 1)[0])
import p7_dev_kaggle_common as C   # noqa: E402

sys.exit(C.main('gpdl'))
