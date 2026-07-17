try:
    import sqlite3  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "deeprank_gnn requires a Python interpreter compiled with sqlite3 support "
    ) from e
