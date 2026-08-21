# deeprank-gnn-esm

Graph Network for protein-protein interface including language model features.

![PyPI - Downloads](https://img.shields.io/pypi/dm/deeprank-gnn-esm)
![PyPI - Version](https://img.shields.io/pypi/v/deeprank-gnn-esm)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deeprank-gnn-esm)
![PyPI - License](https://img.shields.io/pypi/l/deeprank-gnn-esm)

[![ci](https://github.com/haddocking/deeprank-gnn-esm/actions/workflows/ci.yml/badge.svg)](https://github.com/haddocking/deeprank-gnn-esm/actions/workflows/ci.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0021542ee679422ea804cab5f74f724d)](https://app.codacy.com/gh/haddocking/deeprank-gnn-esm/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


For details refer to our publication at <https://academic.oup.com/bioinformaticsadvances/article/4/1/vbad191/7511844>

For detailed protocol to use our `deeprank-gnn-esm` software, refer to our publication
at <https://arxiv.org/abs/2407.16375>

## Installation

```bash
pip install deeprank-gnn-esm
```

### Requirements

Your Python interpreter must be compiled with `sqlite3` support (needed by `pdb2sql`
for structure interface calculations). Most system Python installs already include
this; check with::

```bash
python -c "import sqlite3"
```

If this raises `ModuleNotFoundError: No module named '_sqlite3'`, rebuild or
reinstall Python with `sqlite3` support.

### CPU only

To avoid downloading the heavy CUDA libraries (~3GB), install the CPU-only `torch` first:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install deeprank-gnn-esm
```

### GPU support

GPU support is included automatically — the default PyPI `torch` wheel bundles CUDA.
If your system requires a specific CUDA version, install `torch` first:

```bash
# example for CUDA 12.1
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
pip install deeprank-gnn-esm
```

Check [pytorch.org](https://pytorch.org/get-started/locally/) for the right CUDA version for your system.

## Usage

### As a scoring function

We provide a command-line interface for `deeprank-gnn-esm` that can easily be
used to score protein-protein complexes. It accepts one or more PDB files
(each optionally a multi-model ensemble) in a single invocation. Every pair
of chains in each input structure is scored as a separate interface - a
3-chain complex produces 3 pairwise predictions (A-B, A-C, B-C), an ensemble
of N models produces predictions for every model. The command-line interface
can be used as follows:

```bash
$ deeprank-gnn-esm-predict -h
usage: deeprank-gnn-esm-predict [-h] [--num_cores NUM_CORES] [--output-dir OUTPUT_DIR]
                                 pdb_files [pdb_files ...]

positional arguments:
  pdb_files             Path(s) to the PDB file(s).

optional arguments:
  -h, --help            show this help message and exit
  --num_cores NUM_CORES
                        Number of cores to use (default: 1)
  --output-dir OUTPUT_DIR
                        Directory to save intermediate files in (default: a
                        temporary directory that is discarded after the run).
```

Example, score the `1B6C` complex

```bash
# download it
$ wget https://files.rcsb.org/view/1B6C.pdb -q

$ deeprank-gnn-esm-predict 1B6C.pdb
 2026-07-22 06:08:21,889 predict:41 INFO - Setting up workspace - /tmp/tmpabcd1234
 2026-07-22 06:08:21,945 input:49 INFO - Renumbering structure 1B6C.
 2026-07-22 06:08:22,294 input:99 INFO - Reading sequence of structure 1B6C
 2026-07-22 06:08:22,423 sequence:57 INFO - Generating embeddings for 2 unique sequence(s).
 2026-07-22 06:08:32,459 input:212 INFO - Wrote 1 chain-pair PDB(s) of structure 1B6C to /tmp/tmpabcd1234/structures
 2026-07-22 06:08:36,470 predict:48 INFO - Generating graph, using 1 processors
 2026-07-22 06:09:03,345 predict:63 INFO - Graph file generated: /tmp/tmpabcd1234/graph.hdf5
 2026-07-22 06:09:03,345 predict:69 INFO - Predicting fnat of protein complex.
 2026-07-22 06:09:03,345 predict:77 INFO - Using device: cuda:0
 # ...
 2026-07-22 06:09:07,794 predict:130 INFO - Predicted fnat for 1B6C between chain A and chain B: 0.359
 2026-07-22 06:09:07,803 predict:141 INFO - Output written to /tmp/tmpabcd1234/GNN_esm_prediction.csv
 2026-07-22 06:09:07,805 main:71 INFO - Result saved to /home/deeprank-gnn-esm/GNN_esm_prediction.csv
```

From the output above you can see that the predicted fnat for the 1B6C
complex is **0.359**, this information is also written to the
`GNN_esm_prediction.csv` file in the directory you ran the command from:

```text
pdb_id,chain_i,chain_j,predicted_fnat
1B6C,A,B,0.359
```

By default all intermediate files (renumbered PDB, per-chain embeddings,
per-pair PDBs, graph/prediction hdf5) live in a temporary directory that's
removed once the run finishes. Pass `--output-dir` to keep them around for
inspection instead:

```bash
$ deeprank-gnn-esm-predict 1B6C.pdb --output-dir 1B6C-gnn_esm_pred
```

```text
1B6C-gnn_esm_pred
├── 1B6C.pdb                   #renumbered copy of the input pdb file
├── 1B6C.A.pt                  #esm-2 embedding for chain A in protein 1B6C
├── 1B6C.B.pt                  #esm-2 embedding for chain B in protein 1B6C
├── structures/
│   └── 1B6C_A-B.pdb           #2-chain pdb materialized for the A-B interface
├── graph.hdf5                 #input protein graph in hdf5 format
├── GNN_esm_prediction.hdf5    #prediction output in hdf5 format
└── GNN_esm_prediction.csv     #prediction output in csv format
```

Multiple PDB files (including multi-model ensembles) can be scored in one
call - each input must have a unique filename stem:

```bash
$ deeprank-gnn-esm-predict 1B6C.pdb ensemble.pdb --num_cores 4
```

### As a framework

### Note about input pdb files

To ensure the mapping between interface residue and esm-2 embeddings is correct,
make sure that for all the chains, residue numbering in the PDB file is
continuous and starts with residue '1'.

We provide a script (`scripts/pdb_renumber.py`) to do the numbering.

#### Generate esm-2 embeddings for your protein

- To generate fasta sequences from PDBs, use script `get_fasta.py`

  ```bash
  usage: get_fasta.py [-h] pdb_file_path chain_id1 chain_id2

  positional arguments:
    pdb_file_path  Path to the directory containing PDB files
    chain_id1      Chain ID for the first sequence
    chain_id2      Chain ID for the second sequence

  options:
    -h, --help         show this help message and exit


  python scripts/get_fasta.py tests/data/pdb/1ATN/ A B

  ```

- Generate embeddings in bulk from combined fasta files, use the script
  provided inside esm-2 package,

  ```bash
  $ python esm_2_installation_location/scripts/extract.py \
      esm2_t33_650M_UR50D \
      all.fasta \
      tests/data/embedding/1ATN/ \
      --repr_layers 0 32 33 \
      --include mean per_tok
  ```

  Replace 'esm_2_installation_location' with your installation location,
  'all.fasta' with fasta sequence generated above,
  'tests/data/embedding/1ATN/' with the output folder name for esm embeddings

#### Generate graph

- Example code to generate residue graphs in hdf5 format:

  ```python
  from deeprank_gnn.GraphGenMP import GraphHDF5

  pdb_path = "tests/data/pdb/1ATN/"
  pssm_path = "tests/data/pssm/1ATN/"
  embedding_path = "tests/data/embedding/1ATN/"
  nproc = 20
  outfile = "1ATN_residue.hdf5"

  GraphHDF5(
      pdb_path = pdb_path,
      pssm_path = pssm_path,
      embedding_path = embedding_path,
      graph_type = "residue",
      outfile = outfile,
      nproc = nproc,    #number of cores to use
      tmpdir="./tmpdir")
  ```

- Example code to add continuous or binary targets to the hdf5 file

  ```python
  import h5py
  import random

  hdf5_file = h5py.File('1ATN_residue.hdf5', "r+")
  for mol in hdf5_file.keys():
      fnat = random.random()
      bin_class = [1 if fnat > 0.3 else 0]
      hdf5_file.create_dataset(f"/{mol}/score/binclass", data=bin_class)
      hdf5_file.create_dataset(f"/{mol}/score/fnat", data=fnat)
  hdf5_file.close()
  ```

#### Use pre-trained models to predict

- Example code to use pre-trained deeprank-gnn-esm model

  ```python
  from deeprank_gnn.ginet import GINet
  from deeprank_gnn.NeuralNet import NeuralNet

  database_test = "1ATN_residue.hdf5"
  gnn = GINet
  target = "fnat"
  edge_attr = ["dist"]
  threshold = 0.3
  pretrained_model = 'deeprank-GNN-esm/paper_pretrained_models/scoring_of_docking_models/gnn_esm/treg_yfnat_b64_e20_lr0.001_foldall_esm.pth.tar'
  node_feature = ["type", "polarity", "bsa", "charge", "embedding"]
  device_name = "cuda:0"
  num_workers = 10

  model = NeuralNet(
      database_test,
      gnn,
      device_name = device_name,
      edge_feature = edge_attr,
      node_feature = node_feature,
      target = target,
      num_workers = num_workers,
      pretrained_model = pretrained_model,
      threshold = threshold)

  model.test(hdf5 = "tmpdir/GNN_esm_prediction.hdf5")
  ```
