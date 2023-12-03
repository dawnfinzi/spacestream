"""
Base Fitter class for fitting linear readout layer
(highly inspired by/credit to: https://github.com/neuroailab/VisualCheese) 
"""

import os
import pickle
import sys
from collections import defaultdict

from sklearn.decomposition import PCA

from spacestream.core.feature_extractor import get_features_from_layer
from spacestream.core.paths import RESULTS_PATH
from spacestream.utils.get_utils import get_model


class BaseFitter:
    def __init__(
        self,
        dataloader,
        model_name,
        trained=True,
        spatial_weight=0.5,
        model_seed=0,
        train_frac=None,
        num_train_test_splits=10,
    ):
        self.train_frac = train_frac
        self.num_train_test_splits = num_train_test_splits
        self.dataloader = dataloader

        # self.model_name: e.g., "spacetorch"
        self.model_name = model_name
        self.model = get_model(model_name, trained, spatial_weight, model_seed)

    def get_features(self, layer_name, unit_idx=None):
        features = get_features_from_layer(
            self.model,
            self.dataloader,
            layer_name,
            two_pathway=False,
            reduction_list=None,
            batch_size=128,
            vectorize=True,
        )

        if unit_idx is None:
            return features[layer_name]
        else:
            return features[layer_name][:, unit_idx]  # subselect units

    def train_test_split(self, num_stimuli):
        """
        The implementation should take in a single argument called num_stimuli
        and return a list of dictionaries of length self.num_train_test_splits.
        Each dictionary is of the form {"train": train_indices, "test": test_indices}
        """
        raise NotImplementedError

    def do_pca(self, train_set, test_set):
        print("Performing PCA...")
        assert train_set.shape[1] == test_set.shape[1]
        n_components = 1000

        # If number of features is less than n_components, then do not do PCA.
        if n_components >= train_set.shape[1]:
            return train_set, test_set

        # Do PCA to project features in to n_components dimensions.
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(train_set)
        train_proj = pca.transform(train_set)
        test_proj = pca.transform(test_set)

        return train_proj, test_proj

    def _checker(self, features, labels, layer_name):
        if isinstance(features, list):
            for feat in features:
                print(f"Image features of {layer_name} are of dimensions {feat.shape}.")
                assert feat.ndim == 2
        else:
            print(f"Image features of {layer_name} are of dimensions {features.shape}.")
            assert features.ndim == 2

        # First, obtain the train and test set features,
        if isinstance(features, list):
            num_stimuli = features[0].shape[0]
            for feat in features:
                assert num_stimuli == feat.shape[0]
            if labels is not None:
                assert isinstance(labels, list)
                for label in labels:
                    assert label.shape[0] == num_stimuli
        else:
            num_stimuli = features.shape[0]
            if labels is not None:
                assert labels.shape[0] == num_stimuli
        return num_stimuli

    def fit(
        self,
        task_type,
        layer_name,
        spatial_weight=0.5,
        model_seed=0,
        hemi="rh",
        subj="subj01",
        stream=None,
        unit_idx=None,
        pca=True,
        sampling=0,
        var_splits=0,
    ):
        """
        Returns a metrics dict where the keys are the metrics and the values are
        a list of length self.num_train_test_splits.
        """
        features = self.get_features(layer_name, unit_idx)
        labels = None
        if isinstance(features, tuple):
            assert len(features) == 2  # features and labels
            features, labels = features

        num_stimuli = self._checker(
            features=features, labels=labels, layer_name=layer_name
        )

        splits = self.train_test_split(num_stimuli)

        # Check if the train indicies are identical across all splits.
        duplicate_train = all(
            set(sp["train"]) == set(splits[0]["train"]) for sp in splits
        )

        cls_kwargs = None
        metrics = defaultdict(list)
        for sp_idx, sp in enumerate(splits):
            print(f"Split {sp_idx+1}/{len(splits)}")

            train_idx = sp["train"]
            test_idx = sp["test"]
            if isinstance(features, list):
                train_set = features[sp_idx][train_idx, :]
                test_set = features[sp_idx][test_idx, :]
            else:
                train_set = features[train_idx, :]
                test_set = features[test_idx, :]

            # Second, perform PCA on the train set features and project both
            # sets of features into lower-dimensional space.
            if pca:
                train_set, test_set = self.do_pca(train_set, test_set)
                print(
                    f"Train set projected features are of dimensions {train_set.shape}."
                )
                print(
                    f"Test set projected features are of dimensions {test_set.shape}."
                )

            split_data = {
                "train_features": train_set,
                "train_idx": train_idx,
                "test_features": test_set,
                "test_idx": test_idx,
            }
            if labels is not None:
                split_data["train_labels"] = (
                    labels[sp_idx][train_idx]
                    if isinstance(labels, list)
                    else labels[train_idx]
                )
                split_data["test_labels"] = (
                    labels[sp_idx][test_idx]
                    if isinstance(labels, list)
                    else labels[test_idx]
                )

            # Finally, do fitting on the specific task type.
            if duplicate_train and sp_idx > 0:
                assert cls_kwargs is not None
                # If all the train indices are identical, as indicated by duplicate_train,
                # then avoid rerunning CV on the same train indices by supplying the
                # previous classifier/regression keyword arguments.
                metrics_sp = self.fit_task(task_type, split_data, cls_kwargs=cls_kwargs)
            else:
                metrics_sp = self.fit_task(task_type, split_data)
                cls_kwargs = metrics_sp["cls_kwargs"]

            # Record all the metrics
            for k in metrics_sp.keys():
                metrics[k].append(metrics_sp[k])

        self.save_metrics(
            task_type,
            metrics,
            layer_name,
            spatial_weight,
            model_seed,
            hemi,
            subj,
            stream,
            pca,
            sampling,
            var_splits,
        )

    def fit_task(self, task_type, split_data, cls_kwargs=None):
        raise NotImplementedError

    def save_metrics(
        self,
        task_type,
        metrics,
        layer_name,
        spatial_weight,
        model_seed,
        hemi,
        subj,
        stream,
        pca,
        sampling,
        var_splits,
    ):
        dataset_name = self.dataloader.dataset.get_name()

        sw = "sw" + str(spatial_weight)

        save_dir = os.path.join(
            RESULTS_PATH,
            "analyses/transfer/HVM",
            f"{dataset_name}/{self.model_name}/{layer_name}/{sw}_seed{model_seed}/{subj}",
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        pca_flag = "_no_pca" if not pca else ""
        sampling_flag = f"_top_{str(sampling)}k" if sampling > 0 else ""
        var_splits_flag = "_all_var_splits" if var_splits > 0 else ""
        if stream:
            context = f"{hemi}_{stream}_stream_{task_type}{pca_flag}{sampling_flag}{var_splits_flag}_save_preds.pkl"
        else:
            context = f"{hemi}_all_chosen_units_{task_type}{pca_flag}{sampling_flag}{var_splits_flag}.pkl"
        fname = os.path.join(save_dir, context)
        pickle.dump(metrics, open(fname, "wb"))
        print(f"Saved results to {fname}.")
