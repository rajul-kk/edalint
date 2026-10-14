"""P7 pre-freeze test: a static check that nothing under src/edalint_p7 or scripts/p7_*.py reads the N14 evaluation rows except through
`guard.open_test_rows`. The scanner is exercised on synthetic violations too, so it cannot pass vacuously."""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = {'packaged_split_csvs', 'include_test', 'smoke_test_shapes', '_wanted', 'test_splits', '_test_ids', 'is_test_split'}
OPENERS = {'N14Dataset': {'open_train_rows', 'open_test_rows'}, 'read_split_csv': {'train_names'}, 'stage_all': set(), 'stage_design': {'open_test_rows'}}


def scan_source(src: str, filename: str = '<src>'):
    """Violations: references to loader internals that expose the test split, calls that construct/stage/read split files outside their one
    permitted function, and the bare string 'test' or any 'test_N14' string outside open_test_rows. Docstrings are exempt."""
    tree = ast.parse(src, filename)
    out = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.fn = []

        def visit_FunctionDef(self, node):
            self.fn.append(node.name)
            self.generic_visit(node)
            self.fn.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def in_opener(self):
            return 'open_test_rows' in self.fn

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return                                          # docstring / bare string statement
            self.generic_visit(node)

        def visit_Name(self, node):
            if node.id in FORBIDDEN_NAMES and not self.in_opener():
                out.append(f'{filename}:{node.lineno} name {node.id}')

        def visit_Attribute(self, node):
            if node.attr in FORBIDDEN_NAMES and not self.in_opener():
                out.append(f'{filename}:{node.lineno} attribute {node.attr}')
            self.generic_visit(node)

        def visit_alias(self, node):
            if node.name in FORBIDDEN_NAMES and not self.in_opener():
                out.append(f'{filename}:{node.lineno} import {node.name}')

        def visit_keyword(self, node):
            if node.arg in FORBIDDEN_NAMES and not self.in_opener():
                out.append(f'{filename}:{getattr(node.value, "lineno", 0)} keyword {node.arg}')
            self.generic_visit(node)

        def visit_Call(self, node):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in OPENERS and not (set(self.fn) & OPENERS[name]):
                out.append(f'{filename}:{node.lineno} call {name} outside {sorted(OPENERS[name]) or "any function"}')
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, str) and (node.value == 'test' or 'test_N14' in node.value) and not self.in_opener():
                out.append(f'{filename}:{node.lineno} string {node.value!r}')

    V().visit(tree)
    return out


def p7_sources():
    files = sorted((REPO / 'src' / 'edalint_p7').rglob('*.py')) + sorted((REPO / 'scripts').glob('p7_*.py'))
    return [p for p in files if '__pycache__' not in p.parts]


def test_no_p7_code_reads_test_rows_outside_open_test_rows():
    files = p7_sources()
    assert len(files) >= 8
    bad = []
    for p in files:
        bad += scan_source(p.read_text(encoding='utf-8'), p.relative_to(REPO).as_posix())
    assert bad == []


def test_open_test_rows_is_the_only_function_that_builds_the_dataset_on_the_evaluation_split():
    # N14Dataset is referenced inside exactly two functions: the train opener (empty second split file) and open_test_rows
    # (as the default of its `_factory`); the module-level import is not inside a function
    where = []
    for p in p7_sources():
        tree = ast.parse(p.read_text(encoding='utf-8'))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for c in ast.walk(fn):
                if isinstance(c, ast.Name) and c.id == 'N14Dataset':
                    where.append((p.name, fn.name))
    assert sorted(set(where)) == [('data.py', 'open_train_rows'), ('guard.py', 'open_test_rows')]


def test_train_opener_passes_an_empty_second_split_file():
    src = (REPO / 'src' / 'edalint_p7' / 'data.py').read_text(encoding='utf-8')
    assert "empty.write_text('')" in src and 'split_csvs=(str(train_csv_path()), str(empty))' in src


# -- the scanner itself ------------------------------------------------------------------------------------------------------------------
def test_scanner_flags_synthetic_violations():
    assert scan_source("from edalint.datasets.circuitnet import packaged_split_csvs\n")
    assert scan_source("def f():\n    return packaged_split_csvs()[1]\n")
    assert scan_source("def f(ds):\n    return ds.ids('test')\n")
    assert scan_source("def f(ds):\n    return ds.stage_all(include_test=True)\n")
    assert scan_source("def f():\n    return N14Dataset(task='x')\n")
    assert scan_source("def g():\n    return N14Dataset(task='x')\n")
    assert scan_source("def f():\n    return 'a/test_N14.csv'\n")
    assert scan_source("def f(ds):\n    return ds._wanted\n")
    assert scan_source("def f():\n    return read_split_csv('x')\n")


def test_scanner_accepts_the_permitted_forms():
    assert scan_source("def open_test_rows():\n    ds = N14Dataset()\n    return ds.ids('test'), packaged_split_csvs()\n") == []
    assert scan_source("def open_train_rows():\n    return N14Dataset()\n") == []
    assert scan_source("def train_names():\n    return read_split_csv('x')\n") == []
    assert scan_source('def f():\n    """test rows are read elsewhere"""\n    return 1\n') == []
    assert scan_source("def f(ds):\n    return ds.ids('train')\n") == []
