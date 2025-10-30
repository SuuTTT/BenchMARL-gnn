Developer setup: create a local conda environment and install the repo in editable mode
=================================================================

This file contains reproducible steps to create a local conda environment for development and install
the repository in editable mode.

Minimal recommended steps (fast, no heavy binaries):

1) Create environment from `environment.yml` (runs `pip install -e .` inside env):

   conda env create -f environment.yml

   or, to update an existing env:

   conda env update -f environment.yml --prune

2) Activate the env:

   # on Linux/macOS
   source $(conda info --base)/etc/profile.d/conda.sh
   conda activate benchmarl-dev

3) Notes about heavy dependencies

- The package `benchmarl` requires `torchrl` and `torchvision` (and ultimately `torch`). These are
  binary packages that must be installed according to your machine's CUDA/runtime. Examples:

  # CPU-only (example)
  conda install -y -c pytorch -c conda-forge pytorch torchvision cpuonly

  # CUDA 12.1 (example; pick the right cuda version for your machine)
  conda install -y -c pytorch -c conda-forge pytorch torchvision pytorch-cuda=12.1

  After installing torch/torchvision, install torchrl as per its instructions. For many setups:
  pip install "torchrl>=0.8,<0.11"

4) Optional extras

- If you want VMAS support:
  pip install "vmas>=1.3.4"

- For GNN support (torch_geometric), follow the official PyG install guide for your PyTorch/CUDA
  combination: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

5) Verify editable install

After activating the env, confirm the package is installed in editable mode (should point to repo path):

  pip show benchmarl

You should see "Location: /path/to/BenchMARL-gnn" and "Editable project" behavior when you modify code.

Troubleshooting
- If you hit errors installing `av`, you may need system libraries (ffmpeg). On Ubuntu:
  sudo apt-get install -y libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

Summary
- `environment.yml` does a minimal setup (Python 3.10, pip, installs `-e .`), and you then install
  heavy binary dependencies (PyTorch/torchvision/torchrl/torch_geometric) tailored to your machine.
