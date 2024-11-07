# "A single computational objective drives specialization of streams in visual cortex"

## Authors
* Dawn Finzi :email:
* Eshed Margalit
* Kendrick Kay*
* Daniel L. K. Yamins*
* Kalanit Grill-Spector*

*co-senior authors

## Overview of the repository
This repository has the following components:
* `notebooks/`: Jupyter notebooks (saved as Markdown files) that reproduce all figures and statistics in the paper
* `matlab/`: Matlab code used to generate a reorganized, revised version of Fig 2
* `scripts/`: standalone scripts to be run from the command line. Includes example scripts to run the mapping algorithm (`fitting_one_to_one_unit2voxel.py` and `fitting_one_to_one_voxel2voxel.py` among others).
* `spacestream/`: installable Python package with functionality including model feature extraction, model-to-brain mapping methods, task transfer, spatial and functional metrics of correspondence to the brain, and more.

## Getting started
To install the repository, follow these steps:
1. Create a virtual environment for this project (for example): `python3.7 -m venv spacestream_env`
2. Activate the environment: `source spacestream_env/bin/activate`
3. Clone the repository: `git clone https://github.com/dawnfinzi/spacestream.git`
4. Change your working directory: `cd spacestream`
5. Install the required packages: `pip install -e .`