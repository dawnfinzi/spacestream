"""
Feature Extractor, originally inspired by VisualCheese and SpaceTorch (NeuroAI Lab) and Model Tools
PytorchWrapper (DiCarlo lab)
"""
import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class FeatureExtractor:
    """
    Extracts activations from a layer of a model.
    Arguments:
        dataloader : (torch.utils.data.DataLoader) dataloader. assumes images
                     have been transformed correctly (i.e. ToTensor(),
                     Normalize(), Resize(), etc.)
        two_pathway: (boolean) using a two pathway model (like SlowFast) or not
        n_batches  : (int) number of batches to obtain image features
        vectorize  : (boolean) whether to convert layer features into vector
        device     : (str) can specify device, else defaults to cuda if available
    """

    def __init__(
        self,
        dataloader: DataLoader,
        two_pathway: bool = False,
        n_batches: Optional[int] = None,
        vectorize: bool = False,
        verbose: bool = True,
        device: Optional[str] = None,
    ):
        self.dataloader = dataloader
        self.two_pathway = two_pathway
        self.n_batches = n_batches or len(self.dataloader)
        self.vectorize = vectorize
        self.verbose = verbose
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _resolve_sequential_module_from_str(
        self, model: nn.Module, model_layer: str
    ) -> nn.Module:
        """
        Recursively resolves the model layer name by drilling into nested nn.Sequential
        """
        # initialize top level as the model
        layer_module = model

        # iterate over parts separated by a period, replacing layer_module with the next
        # sublayer in the chain
        for part in model_layer.split("."):
            layer_module = layer_module._modules.get(part)
            assert (
                layer_module is not None
            ), f"No submodule found for layer {model_layer}, at part {part}."

        return layer_module

    def get_activation(self, name):
        def hook(model, input, output):
            # If output is a tuple/list, prefer the first element
            if isinstance(output, (tuple, list)) and len(output) > 0:
                out = output[0]
            # If it's a HF-style output with `last_hidden_state`, use that
            elif hasattr(output, "last_hidden_state"):
                out = output.last_hidden_state
            # fallback: keep original
            else:
                out = output
            out = out.cpu().numpy()
            self.layer_batch[name].append(out)

        return hook

    def extract_features(
        self,
        model: nn.Module,
        model_layer_strings: Union[str, List[str]],
        time_step: Optional[int] = None,
        reduction_list: Optional[Union[int, List[int]]] = None,
        return_inputs: bool = False,
        return_inputs_and_labels: bool = False,
    ):

        if not isinstance(model_layer_strings, list):
            model_layer_strings = [model_layer_strings]

        assert len(model_layer_strings), "no layer strings provided"

        if (
            return_inputs_and_labels and self.two_pathway
        ):  # currently only tested for sine gratings dataset
            assert "Not implemented yet"
        if (
            return_inputs and return_inputs_and_labels
        ):  # return inputs and labels if both flags specified
            return_inputs = False

        self.layer_results: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {
            k: [] for k in model_layer_strings
        }

        # switch model to eval mode
        if self.device == "cuda":
            model.cuda().eval()
        else:
            model.cpu().eval()

        self.inputs = list()
        self.labels = list()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.dataloader),
                total=self.n_batches,
                desc="batch",
                disable=not self.verbose,
            ):

                self.layer_batch: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {
                    k: [] for k in model_layer_strings
                }

                if batch_idx == self.n_batches:
                    break

                if self.two_pathway:
                    x = [i.to(self.device)[None, ...] for i in batch]
                    x = [torch.squeeze(i, axis=0) for i in x]
                else:
                    if type(self.dataloader.dataset[0]) is tuple:
                        x, label_x = batch
                        x = x.to(self.device)
                    else:
                        x = batch.to(self.device)

                # add forward hooks to each model layer
                hooks = []
                for lix, layer_name in enumerate(model_layer_strings):
                    layer = self._resolve_sequential_module_from_str(model, layer_name)
                    hook = layer.register_forward_hook(self.get_activation(layer_name))
                    hooks.append(hook)

                _ = model(x)

                if return_inputs or return_inputs_and_labels:
                    if self.two_pathway:
                        self.inputs.append(
                            x[0].cpu().numpy()
                        )  # just return slow pathway inputs
                    else:
                        self.inputs.append(x.cpu().numpy())

                if return_inputs_and_labels:
                    self.labels.append(label_x.cpu().numpy())

                # Reset forward hook so next time function runs, previous hooks
                # are removed
                for hook in hooks:
                    hook.remove()

                for k, v in self.layer_batch.items():
                    if time_step is None:
                        v = v[0]
                    else:
                        v = v[time_step]
                    self.layer_results[k].append(v)

        if return_inputs or return_inputs_and_labels:
            self.inputs = np.concatenate(self.inputs)

        if return_inputs_and_labels:
            self.labels = np.concatenate(self.labels)

        self.layer_feats = {k: np.concatenate(v) for k, v in self.layer_results.items()}

        for lix, layer_name in enumerate(model_layer_strings):
            if reduction_list and reduction_list[lix] > 0:
                self.layer_feats[layer_name] = self.layer_feats[layer_name].mean(
                    axis=reduction_list[lix]
                )
            if self.vectorize:
                self.layer_feats[layer_name] = np.reshape(
                    self.layer_feats[layer_name],
                    (len(self.layer_feats[layer_name]), -1),
                )

        if return_inputs:
            return self.layer_feats, self.inputs
        elif return_inputs_and_labels:
            return self.layer_feats, self.inputs, self.labels

        return self.layer_feats


def get_features_from_layer(
    model: nn.Module,
    dataloader: DataLoader,
    model_layer_strings: Union[str, List[str]],
    time_step: Optional[int] = None,
    two_pathway: bool = False,
    reduction_list: Optional[
        Union[int, List[int]]
    ] = None,  # temporal dimension to average over
    batch_size: int = 32,
    max_batches: Optional[int] = None,
    return_inputs: bool = False,
    return_inputs_and_labels: bool = False,
    vectorize: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:

    n_images: int = len(dataloader.dataset)
    n_batches: int = max_batches or math.ceil(n_images / batch_size)

    extractor = FeatureExtractor(
        dataloader, two_pathway, n_batches, vectorize=vectorize, device=device
    )
    return extractor.extract_features(
        model,
        model_layer_strings,
        time_step,
        reduction_list,
        return_inputs=return_inputs,
        return_inputs_and_labels=return_inputs_and_labels,
    )
