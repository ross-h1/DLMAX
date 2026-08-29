# Bundled example datasets

Small datasets shipped with DLMAX for the quickstart and documentation
examples. Loaded via :func:`DLMAX.datasets.load_dataset`.

## Convention

Drop a CSV here named `<dataset_name>.csv`, in **long format** with exactly
these columns:

| column      | meaning                                             |
|-------------|-----------------------------------------------------|
| `unique_id` | series identifier (one value per series)            |
| `ds`        | timestamp (date string) or integer step index       |
| `y`         | observed value                                      |

This is exactly the input `AutoFFS.fit` / `AutoFFS.forecast` expect, so a
loaded dataset can be passed straight through. `load_dataset(name)` reads
`<name>.csv`, and `available_datasets()` lists every `*.csv` here by stem.

Keep these small — they ship inside the wheel.
