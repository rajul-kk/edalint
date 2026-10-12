"""CircuitNet-N28 loader STUB. The data was not obtained (Google Drive quota; see data_cards.md section 3).

Expected layout, from the authors' scripts: per-sample `.npy` files with the same feature names as N14 and the same 256 x 256
preprocessing; official design-level split train_N28.csv (RISCY, RISCY-FPU) / test_N28.csv (zero-riscy). Nothing is read here.
"""
from __future__ import annotations


class N28Dataset:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('CircuitNet-N28 data has not been obtained; this is a loader stub (see data_cards.md section 3).')
