"""
Regression utils adapted
(credit to Yamins lab: https://github.com/neuroailab/)
"""

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV

from spacestream.utils.general_utils import featurewise_norm, rsquared


# very trimmed down get_splits function
def get_splits(
    data,
    split_index,
    num_splits,
    num_per_class_test,
    num_per_class_train,
    exclude=None,
    seed=0,
):
    """
    construct a consistent set of splits for cross validation

    arguments:
        data: numpy array data to split
        split_index: index to split the data over
        num_per_class_test: number of testing examples for each unique
                            split_by category
        num_per_class_train: number of train examples for each unique
                            split_by category
        exlude: mask of indices not to include in splits
        seed: seed for random number generator
    """

    # get all potential train and test indices
    if exclude is None:
        train_inds = np.arange(np.size(data, axis=split_index))
        test_inds = np.arange(np.size(data, axis=split_index))
    else:
        train_inds = np.arange(np.size(data, axis=split_index))[~exclude]
        test_inds = np.arange(np.size(data, axis=split_index))[~exclude]

    # seed the random number generator
    rng = np.random.RandomState(seed=seed)

    # construct the splits one by one
    splits = []
    for _split_ind in range(num_splits):
        # first construct the testing data
        actual_test_inds = []
        tidx = np.arange(len(test_inds))
        perm = rng.permutation(tidx)
        actual_test_inds = np.sort(test_inds[perm[:num_per_class_test]])

        # now, since the pools of possible train and test data overlap,
        # but since we don't want the actual train and data examples to overlap at all,
        # remove the chosen test examples for this split from the pool of possible
        # train examples for this split
        remaining_available_train_inds = np.unique(
            list(set(train_inds).difference(actual_test_inds))
        )

        # first construct training the same way
        actual_train_inds = []
        num_possible_train_inds = len(remaining_available_train_inds)
        perm = rng.permutation(num_possible_train_inds)
        actual_train_inds = np.sort(
            remaining_available_train_inds[perm[:num_per_class_train]]
        )

        # check that there's no overlap!!
        assert set(actual_train_inds).intersection(actual_test_inds) == set([])

        split = {"train": actual_train_inds, "test": actual_test_inds}
        splits.append(split)

    return splits


def train_and_test_scikit_regressor(
    features,
    labels,
    splits,
    model_class,
    model_args=None,
    gridcv_params=None,
    gridcv_args=None,
    fit_args=None,
    feature_norm=True,
    return_models=False,
):
    """This function is very similar to the train_and_test_scikit_classifier function
    except it is adapted for working with regressors.
    """

    if model_args is None:
        model_args = {}
    if fit_args is None:
        fit_args = {}

    training_sidedata = []

    models = []
    train_results = []
    test_results = []

    for split in splits:

        # here we instantiate the general regressor whatever it is
        model = model_class(**model_args)
        if gridcv_params is not None:
            if gridcv_args is None:
                gridcv_args = {}
            model = GridSearchCV(model, gridcv_params, **gridcv_args)

        # get the train/test split data
        train_inds = split["train"]
        test_inds = split["test"]
        train_features = features[train_inds]
        train_labels = labels[train_inds]
        test_features = features[test_inds]
        test_labels = labels[test_inds]

        # train the model ...
        if feature_norm:
            train_features, fmean, fvar = featurewise_norm(train_features)
            sidedata = {"fmean": fmean, "fvar": fvar}
            training_sidedata.append(sidedata)
        model.fit(train_features, train_labels, **fit_args)

        # ... and get training predictions and results
        train_predictions = model.predict(train_features)
        train_result = evaluate_regression_results(train_predictions, train_labels)
        train_results.append(train_result)

        # test the model ...
        if feature_norm:
            test_features, _ignore, _ignore = featurewise_norm(
                test_features, fmean=fmean, fvar=fvar
            )
        # ... and get testing predictions and results
        test_predictions = model.predict(test_features)
        test_result = evaluate_regression_results(test_predictions, test_labels)
        test_results.append(test_result)

        if return_models:
            models.append(model)

    # aggregate results over splits
    train_results = aggregate_regression_results(train_results)
    test_results = aggregate_regression_results(test_results)
    results = {
        "train": train_results,
        "test": test_results,
        "training_sidedata": training_sidedata,
    }
    if return_models:
        results["models"] = models

    return results


def evaluate_regression_results(predicted, actual):
    """computing various useful metrics for regression results"""
    result = {}
    if (
        actual.ndim > 1
    ):  # this is triggered if the prediction is of multiple outputs at once
        result["pearson_array"] = np.array(
            [stats.pearsonr(p, a)[0] for p, a in zip(predicted.T, actual.T)]
        )
        result["spearman_array"] = np.array(
            [stats.spearmanr(p, a)[0] for p, a in zip(predicted.T, actual.T)]
        )
        result["rsquared_array"] = np.array(
            [rsquared(p, a) for p, a in zip(predicted.T, actual.T)]
        )
        result["pearson"] = np.median(result["pearson_array"])
        result["spearman"] = np.median(result["spearman_array"])
        result["rsquared"] = np.median(result["rsquared_array"])
    else:
        result["pearson"] = stats.pearsonr(predicted, actual)[0]
        result["spearman"] = stats.spearmanr(predicted, actual)[0]
        result["rsquared"] = rsquared(predicted, actual)
    return result


def aggregate_regression_results(results_by_split):
    """convenience function aggregating results of regression tests over data splits"""
    results = {}
    results["by_split"] = results_by_split
    ns = len(results_by_split)
    for k in results_by_split[0]:
        arr = np.array([results_by_split[i][k] for i in range(ns)])
        if arr.ndim == 1:
            results["mean_" + k] = arr.mean()
            results["std_" + k] = arr.std()
        else:
            results["mean_" + k] = arr.mean(axis=0)
            results["std_" + k] = arr.std(axis=0)
    return results
