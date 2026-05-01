# Present so `pytest -q` works from the repo root: pytest inserts the directory
# containing the rootdir conftest onto sys.path, making the `src` package importable
# from the test modules (which do `from src... import ...`). Without it, pytest's
# default "prepend" import mode only adds tests/ to sys.path and `src` is not found.
