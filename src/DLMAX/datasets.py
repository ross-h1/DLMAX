"""Bundled example datasets for the DLMAX quickstart and documentation.

Each dataset is a small CSV shipped inside the package, in long format with
columns ``unique_id``, ``ds``, ``y`` — exactly the input ``AutoFFS.fit`` /
``AutoFFS.forecast`` expect, so a loaded dataset can be passed straight
through:

>>> from DLMAX import AutoFFS
>>> from DLMAX.datasets import load_dataset, available_datasets
>>> available_datasets()
['airline_passengers']
>>> df = load_dataset("airline_passengers")
>>> df.shape
(144, 3)
>>> fc = AutoFFS(season_length=12).forecast(df, h=12)   # doctest: +SKIP
"""

from importlib.resources import files

import pandas as pd

__all__ = ["available_datasets", "load_dataset"]

_DATA_DIR = files("DLMAX") / "data"
_REQUIRED_COLUMNS = ("unique_id", "ds", "y")


def available_datasets() -> list[str]:
    """List the names of the bundled example datasets.

    Returns
    -------
    list of str
        Dataset names (CSV stems), sorted alphabetically. Pass any of these
        to :func:`load_dataset`.
    """
    return sorted(
        p.name[:-4] for p in _DATA_DIR.iterdir() if p.name.endswith(".csv")
    )


def load_dataset(name: str) -> pd.DataFrame:
    """Load a bundled example dataset as a long-format DataFrame.

    Parameters
    ----------
    name : str
        Dataset name without extension; see :func:`available_datasets`.

    Returns
    -------
    pandas.DataFrame
        Columns ``unique_id``, ``ds``, ``y``. ``ds`` is returned as integers
        when the source is integer-indexed, otherwise parsed to datetime.
        Ready to pass to ``AutoFFS.fit`` / ``AutoFFS.forecast``.

    Raises
    ------
    FileNotFoundError
        If ``name`` is not a bundled dataset.
    """
    path = _DATA_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"No bundled dataset {name!r}. "
            f"Available: {available_datasets()}."
        )

    with path.open("r") as fh:
        df = pd.read_csv(fh)

    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset {name!r} is missing required columns {sorted(missing)}; "
            f"expected {list(_REQUIRED_COLUMNS)}."
        )

    # Integer-indexed series keep an integer ``ds``; everything else is parsed
    # to datetime, mirroring AutoFFS's own input handling.
    if not pd.api.types.is_integer_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])

    return df.loc[:, list(_REQUIRED_COLUMNS)]
