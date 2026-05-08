#!/usr/bin/env bash

set -euo pipefail

pip install -e .

pip install natsort
# These source builds need access to packages from the active conda env.
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
pip install smplx==0.1.28

pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
# pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
