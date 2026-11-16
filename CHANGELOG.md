# Changelog

## v2.0.1

- Chain pairs without interface are reported as `NA` instead of crashing the run; `GNN_esm_prediction_summary.csv` is returned, with the mean, standard deviation and minimum fnat per model; waters, ions and ligands are removed from the input, as they changed the ESM embeddings and thus the predictions ([#38](https://github.com/haddocking/deeprank-gnn-esm/issues/38)).

## v2.0.2

- Divide the command line interface entry point `main()` into two funtions: `main()` -> `_main()`. The first one (`main()`) reads the command line arguments from the user, and gives, use argparse to load it, and send it to the `_main()` function. This small change allows to use the `_main()` as library without the need of `sys.argv` arguments present. For users, nothing should change and those changes should not affect in any manner.

