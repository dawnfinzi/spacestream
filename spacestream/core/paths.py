"""
Path management
"""
"""
oakzfs = "/sni-storage/kalanit/Projects/Dawn/NSD/"
oaknative = "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/"
local = "/home/dfinzi/NSD/"

stem = oaknative 

CODE_PATH = local + "code/fit_pipeline/"
UTILS_PATH = local + "code/fit_pipeline/utils/"
DATALOADER_PATH = stem + "code/fit_pipeline/datasets/"

LOCALDATA_PATH = stem + "local_data/"
BETA_PATH = LOCALDATA_PATH + "processed/organized_betas/"
STIM_PATH = stem + "data/nsddata_stimuli/stimuli/nsd/"
NSDDATA_PATH = stem + "data/nsddata/"
FS_PATH = stem + "local_data/freesurfer/"
FEATS_PATH = stem + "results/models/features/"
RESULTS_PATH = stem + "results/"
SPACETORCH_PATH = stem + "results/spacetorch"

HVM_PATH = "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/hvm/"
SINE_GRATING_PATH = (
    "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/sine_grating_images_20190507"
)
"""

DATA_PATH = "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/streamlined/data/"
RESULTS_PATH = (
    "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/streamlined/results/"
)

# not included in general release due to size, inquire for access
BETA_PATH = "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/local_data/processed/organized_betas/"
STIM_PATH = "/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata_stimuli/stimuli/nsd/"  # available through NSD release
HVM_PATH = "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/hvm/"
SINE_GRATING_PATH = (
    "/oak/stanford/groups/kalanit/biac2/kgs/ml_datasets/sine_grating_images_20190507"
)
