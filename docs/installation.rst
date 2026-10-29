Installation
=========================

deeprank-gnn-esm requires Python 3.9 to 3.14. Your Python interpreter must be
compiled with ``sqlite3`` support (needed by ``pdb2sql``); check with:

``python3 -c "import sqlite3"``

If this raises ``ModuleNotFoundError: No module named '_sqlite3'``, rebuild or
reinstall Python with ``sqlite3`` support.

Via Python Package
-----------------------------

We recommend installing deeprank-gnn-esm in a Python virtual environment, to
keep its dependencies separate from other projects:

.. code-block:: bash

  python3 -m venv .venv
  source .venv/bin/activate      # csh/tcsh: source .venv/bin/activate.csh; Windows: .venv\Scripts\activate
  pip install --upgrade pip
  pip install deeprank-gnn-esm

Activate the environment (``source .venv/bin/activate``) in every new shell
before using deeprank-gnn-esm.

CPU only
~~~~~~~~~~~~~

To avoid downloading the heavy CUDA libraries (~3GB), install the CPU-only
``torch`` first, in the activated environment:

.. code-block:: bash

  pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
  pip install deeprank-gnn-esm

GPU support
~~~~~~~~~~~~~

GPU support is included automatically: the default PyPI ``torch`` wheel bundles
CUDA. If your system requires a specific CUDA version, install ``torch`` first:

.. code-block:: bash

  # example for CUDA 12.1
  pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
  pip install deeprank-gnn-esm

Check https://pytorch.org/get-started/locally/ for the right CUDA version for
your system.

Via GitHub
-------------

For users who would like to contribute, the code is hosted on GitHub_.
See ``CONTRIBUTING.md`` in the repository for the development guidelines.

.. _GitHub: https://github.com/haddocking/deeprank-gnn-esm

To install the code:

.. code-block:: bash

  git clone https://github.com/haddocking/deeprank-gnn-esm.git
  cd deeprank-gnn-esm
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e '.[dev]'

You can then test the installation from the repository root:

.. code-block:: bash

  pytest -m "not e2e"   # unit tests
  pytest -m e2e         # end-to-end tests, downloads the ESM-2 model (~2.5 GB) on first use

.. note::
  In case PyTorch Geometric installation fails, refer to the installation guide:
  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
