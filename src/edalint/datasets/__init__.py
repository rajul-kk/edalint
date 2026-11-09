from .circuitnet import N14Dataset
from .folder import FolderDataset
from .iccad23 import ICCAD23Dataset
from .n28 import N28Dataset

__all__ = ['FolderDataset', 'ICCAD23Dataset', 'N14Dataset', 'N28Dataset', 'open_dataset']


def open_dataset(name: str, **kw):
    """Factory used by the CLI. Names: iccad23, n14-ir_drop, n14-drc, n14-congestion (and n28-*: stub), or a folder of .npz files
    (a path, or folder:<path>; see edalint.datasets.folder)."""
    from pathlib import Path
    n = name.lower()
    if name.startswith('folder:') or not (n == 'iccad23' or n.startswith(('n14', 'n28'))) and Path(name).is_dir():
        return FolderDataset(name[len('folder:'):] if name.startswith('folder:') else name,
                             **{k: v for k, v in kw.items() if k != 'root'})
    if n == 'iccad23':
        return ICCAD23Dataset(**{k: v for k, v in kw.items() if k in ('root', 'resize', 'protocol_dir')})
    if n.startswith('n14'):
        task = n.split('-', 1)[1] if '-' in n else 'ir_drop'
        return N14Dataset(task=task, **kw)
    if n.startswith('n28'):
        return N28Dataset()
    raise ValueError(f'unknown dataset {name!r}; use iccad23, n14-{{ir_drop,drc,congestion}} or a folder of .npz files')
