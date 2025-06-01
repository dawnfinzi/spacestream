# "A single computational objective can produce specialization of streams in visual cortex"

## Authors
* Dawn Finzi :email:
* Eshed Margalit
* Kendrick Kay*
* Daniel L. K. Yamins*
* Kalanit Grill-Spector*

*co-senior authors

## Overview of the repository
This repository has the following components:
* `notebooks/`: Jupyter notebooks that reproduce all the statistics in the paper and the relevant figures.
* `matlab/`: Matlab code used to generate Figures 2 & 3, as well as a few preprocessing helper functions.
* `scripts/`: standalone scripts to be run from the command line. Includes example scripts to run the mapping algorithm (`fitting_one_to_one_unit2voxel.py` and `fitting_one_to_one_voxel2voxel.py` among others).
* `spacestream/`: installable Python package with functionality including model feature extraction, model-to-brain mapping methods, task transfer, spatial and functional metrics of correspondence to the brain, and more.

## Getting started
To install the repository, follow these steps:
1. Create a virtual environment for this project (for example): `python3.8 -m venv spacestream_env`
2. Activate the environment: `source spacestream_env/bin/activate`
3. Clone the repository: `git clone https://github.com/dawnfinzi/spacestream.git`
4. Change your working directory: `cd spacestream`
5. Install the required packages: `pip install -e .`

## Where to find relevant data
The data for this paper comes from the [Natural Scenes Dataset](https://naturalscenesdataset.org/). We use the b3 preparation of the betas, aligned to the fsaverage surface. However, we also z-score the betas prior to analysis (code to do so can be found in `matlab/prepare_betas.m`). As this is a computationally intensive step, we share the (large) processed betas in this Google Drive [link](https://drive.google.com/drive/folders/1kiomjmIVbilqfurAJnV-XIhwrhbR9zIq). 

Additional processing steps created intermediate outputs. For example, running `scripts/functional_swap.py` for the MB models created `swapopt_swappedon_sine_gratings.npz`, which contains the optimized model unit positions on the simulated cortical sheet used by `scripts/fitting_one_to_one_unit2voxel.py`. To ease testing and reproduction of results, we have also uploaded these intermediate outputs to an [OSF project](https://osf.io/qy32x/). This project additionally includes all model checkpoints specific to this paper. 

The easiest way to use the data is to create a local folder `$YOUR_FOLDER` where you clone this repo and install the repo as specified above. We then recommend creating data/ and results/ subfolders and downloading the file structure as is from the OSF project into your data subfolder. If using the processed betas, those should be downloaded into `$YOUR_FOLDER/data/brains/organized_betas`. You should then be able to run the scripts included in this repo just by updating the path_stem in `spacestream/core/paths.py` to `$YOUR_FOLDER`.

Note that this repository includes the code to run the iterative one-to-one mapping algorithm with a smoothness constraint introduced in [Topographic DCNNs trained on a single self-supervised task capture the functional organization of cortex into visual processing streams](https://openreview.net/forum?id=E1iY-d13smd) from SVHRM @ NeurIPS 2022. However, "A single computational objective can produce specialization of streams in visual cortex" no longer uses this algorithm and instead reports results from only the simpler initial one-to-one mapping based purely on functional correlations. This mapping is saved as "checkpoint0" from `fitting_one_to_one_unit2voxel.py` and `fitting_one_to_one_voxel2voxel.py`, which is still capable of running the iterative neighborhood algorithm as mentioned, if `--max_iter` is set to > 0. 