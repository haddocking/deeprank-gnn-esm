# Changelog

## Unreleased

- Chain pairs without interface are reported as `NA` instead of crashing the run; `GNN_esm_prediction_summary.csv` is returned, with the mean, standard deviation and minimum fnat per model; waters, ions and ligands are removed from the input, as they changed the ESM embeddings and thus the predictions ([#38](https://github.com/haddocking/deeprank-gnn-esm/issues/38)).
