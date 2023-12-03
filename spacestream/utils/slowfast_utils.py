from collections import OrderedDict

import torch

import spacestream.models as models


def get_slowfast_model(arch_name, trained, **kwargs):
    """
    Inputs:
        arch_name : (string) name of deep net architecture.
        trained   : (boolean) whether or not to load a pretrained model.

    Outputs:
        model     : (torch.nn.Module) model
    """

    try:
        print(f"Loading {arch_name}. Pretrained: {trained}.")

        model = models.__dict__[arch_name](pretrained=trained, **kwargs)
    except:
        raise ValueError(f"{arch_name} not implemented yet.")

    return model


def load_slowfast_model(arch_name, trained=False, model_path=None, **kwargs):
    """
    Inputs:
        arch_name  : (string) name of architecture (e.g. "resnet18")
        trained    : (boolean) whether to load a pretrained or trained model.
        model_path : (string) path of model checkpoint from which to load
                     weights.
    Outputs:
        model      : (torch.nn.Module) model
    """
    model = get_slowfast_model(arch_name, trained=trained, **kwargs)

    # Load weights if params file is given.
    if model_path is not None:
        try:
            if torch.cuda.is_available():
                params = torch.load(model_path, map_location="cuda")
            else:
                params = torch.load(model_path, map_location="cpu")
        except:
            raise ValueError(f"Could not open file: {model_path}")

        if "state_dict" in params.keys():
            sd = params["state_dict"]
        else:
            # Hack to also check to see if the "model_state_dict" key exists
            if "model_state_dict" in params.keys():
                sd = params["model_state_dict"]
            else:
                raise Exception("Cannot find state dict to load weights.")

        # Remove `module.` in the state dictionary of DDP/DP models
        new_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith("module."):
                name = k[7:]  # Remove 'module.' of dataparallel/DDP
            else:
                name = k
            new_sd[name] = v

        model.load_state_dict(new_sd)
        print(f"Loaded parameters from {model_path}")

    # Set model to eval mode
    model.eval()

    return model
