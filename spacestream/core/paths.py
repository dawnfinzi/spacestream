"""
Path management
"""
# Update with your path stem here!
path_stem = "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/streamlined/"

# This data path corresponds to the following project on OSF: https://osf.io/qy32x/files/osfstorage
DATA_PATH = path_stem + "data/"
RESULTS_PATH = path_stem + "results/"

# These files were too large for OSF but can be found in this google drive link:
# https://drive.google.com/drive/folders/1kiomjmIVbilqfurAJnV-XIhwrhbR9zIq
BETA_PATH = path_stem + "data/brains/organized_betas/"

# not included in general release due to size, inquire for access
NSDDATA_PATH = "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/"
STIM_PATH = NSDDATA_PATH + "nsddata_stimuli/stimuli/nsd/"  # available through NSD release
HVM_PATH = "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/hvm/"
SINE_GRATING_PATH = (
    "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/sine_grating_images"
)
