# Releasing edalint

Publishing uses PyPI trusted publishing from `.github/workflows/release.yml`: no API token is stored anywhere. Steps 1-2 are
one-time and need the owner's browser.

| step | where | what |
|---|---|---|
| 1 | test.pypi.org and pypi.org, Account settings, Publishing, "Add a new pending publisher" | project `edalint`, owner `rajul-kk`, repository `edalint`, workflow `release.yml`, environment `testpypi` (TestPyPI) / `pypi` (PyPI) |
| 2 | GitHub, repository Settings, Environments | create `testpypi` and `pypi` (optionally: required reviewer on `pypi`) |
| 3 | `pyproject.toml` and `src/edalint/__init__.py` | the new version, the same in both; note the code-hash move in `protocol/CODE_HASH_LINEAGE.json` (a test enforces it) |
| 4 | git | tag `v<version>` on main and push the tag: the release workflow builds and checks the sdist and wheel and that the tag matches the version |
| 5 | GitHub, Actions, release, "Run workflow" | target `testpypi`; then `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple edalint==<version>` and run `edalint checks` |
| 6 | same | target `pypi` |

The GitHub Action needs no publishing: `uses: rajul-kk/edalint@v<version>` installs edalint from the tagged source.
