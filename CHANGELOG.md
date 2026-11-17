# Changelog

## v2.0.2

- Refactor `main()` to ease the use as a library. Divided the command line interface entry point `main()` into two functions: `main()` -> `run_predictions()`. The first one (`main()`) reads the command line arguments from the user, and gives, use argparse to load it, and send it to the `run_predictions()` function ([#50](https://github.com/haddocking/deeprank-gnn-esm/pull/50)).
- Python 3.9 is no longer supported; deeprank-gnn-esm now requires Python 3.10 to 3.14 ([#50](https://github.com/haddocking/deeprank-gnn-esm/pull/50)).

## v2.0.1

- Chain pairs without interface are reported as `NA` instead of crashing the run; `GNN_esm_prediction_summary.csv` is returned, with the mean, standard deviation and minimum fnat per model; waters, ions and ligands are removed from the input, as they changed the ESM embeddings and thus the predictions ([#38](https://github.com/haddocking/deeprank-gnn-esm/issues/38)).
