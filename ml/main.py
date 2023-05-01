import logging
import os
import timeit
import typing
from dataclasses import dataclass
from math import ceil
from warnings import simplefilter

import numpy as np
import pandas
import pandas as pd
from numpy import floor
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
# todo: find way that br removal doesn't glue two words together
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

from latex import evaluationsToLatex

from formatting import printClassifierLatex, printIterationLatex, ColorConsoleFormatter
from flags import verbose, binary
from util import contextual_resample_comp, get_altset_iloc

# Used for model persistence
from joblib import dump, load

logger = logging.getLogger("ML")
logger.setLevel(logging.DEBUG if verbose else logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColorConsoleFormatter('%(name)s: %(message)s'))
logger.addHandler(handler)


@dataclass
class Vectorization:
    features: np.ndarray
    labels: 'typing.Any'


@dataclass
class Split:
    training: np.ndarray
    testing: np.ndarray

    def split(self, both):
        return both[self.training], both[self.testing]


class Vectorizer:
    """
    Wrapper class for vectorizer, features, labels and division between training and testing
    """
    name: str
    extractor: 'typing.Any'
    features: np.ndarray

    def __init__(self, name, extractor):
        self.name = name
        self.extractor = extractor

    def extract_features(self, corpus):
        """
        Perform feature extraction on a set of strings
        :param corpus: a set of strings, each will be transformed into a feature
        :return: set of features
        """
        X = self.extractor.fit_transform(corpus)
        self.features = X


def evaluate_model(model, x_test, y_true, name):
    """
    Calculate evaluation metrics for a model
    :param model: the trained model to evaluate
    :param x_test: the features to test on
    :param y_true: ground truth: labels manually assigned
    :return: precision, recall, f1-score
    """
    y_pred = []
    for feature in x_test:
        y_pred.append(model.predict(feature)[0])

    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.savefig("confusion/" + name + ".png")

    return (
        precision_score(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro"),
        f1_score(y_true, y_pred, average="macro"),
        y_pred
    )


test_x_sub, test_y_sub = None, None


def batch_train(features, labels, classifier, increase_step, kfold_splits):
    """
    :param features: total 2D array of features
    :param labels: 1D array of labels
    :param classifier: The classifier that is to be used
    :param increase_step: how big a subset should differ compared from previous subset
                (e.g. if 100 and subset.len = 200, the next subset will contain 300 features)
    :param kfold_splits: number of splits done for the kfold
    :return:
    """
    global test_y_sub, test_x_sub  # debugging

    kf = KFold(n_splits=kfold_splits)

    columns = ["training size",
               "avg_precision",
               "avg_recall,",
               "avg_f1",
               "precisions",
               "recalls",
               "f1s",
               ]

    rows = []

    # maybe change this, smaller subsets will be trained more than higher ones
    initial_size = increase_step
    subset_count = ceil(features.shape[0] / increase_step)
    leftover_size = features.shape[0] % increase_step

    for i in range(subset_count):
        subset_size = initial_size + increase_step * i
        if i == subset_count - 1 and leftover_size != 0:
            subset_size = subset_size + leftover_size - increase_step
        x_sub, y_sub = resample(features, labels, n_samples=subset_size)

        precisions, recalls, f1s = [], [], []

        for train_index, test_index in kf.split(x_sub, y_sub):
            x_train = x_sub[train_index]
            y_train = y_sub[train_index]
            x_test = x_sub[test_index]
            y_true = y_sub[test_index]

            # todo: add GridSearch here in place of the model, with as parameter the classifier and parameters to adjust
            model = classifier.fit(x_train, y_train)
            precision, recall, f1, y_pred = evaluate_model(model, x_test, y_true)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        rows.append([subset_size,
                     np.mean(precisions),
                     np.mean(recalls),
                     np.mean(f1s),
                     precisions,
                     recalls,
                     f1s
                     ])

    return pandas.DataFrame(rows, columns=columns), model


debug_split = None


def batch_grid_train(f_train, l_train, f_test, l_test, classifier, increase_step,
                     training_pp: pd.DataFrame, parameters, name, binary):
    """
    :param parameters:
    :param training_pp: containing CONTENT, SUBJECT and LABEL
    :param classifier: The classifier that is to be used
    :param increase_step: how big a subset should differ compared from previous subset
                (e.g. if 100 and subset.len = 200, the next subset will contain 300 features)
    :return:
    """

    columns = ["training size", "aimed size", "precision", "recall", "f1"]
    rows = []

    # 'training pp' used to split training set in 5-fold-like, returns indices which can be used to subset 'features'

    initial_size = increase_step
    subset_count = ceil(f_train.shape[0] / increase_step)
    leftover_size = f_train.shape[0] % increase_step

    progress = tqdm(range(subset_count))
    for i in progress:
        subset_size = initial_size + increase_step * i
        if i == subset_count - 1 and leftover_size != 0:
            subset_size = subset_size + leftover_size - increase_step

        progress.set_description(f"Training approximate size of {subset_size}")

        # # get subset of approximately given size
        ix_sub = contextual_resample_comp(training_pp, subset_size)
        # x_sub, y_sub = f_train[ix_sub], l_train[ix_sub]
        # sub_pp = training_pp.iloc[ix_sub]

        # # k-fold like splitting
        # # not actual k-fold: no guarantee that each fold is completely separate
        # splits = []
        # split_size = floor(len(ix_sub) / 5)
        # logger.debug(f"Creating splits of {split_size}")

        # for _ in range(5):
        #     test_index = contextual_resample_comp(sub_pp, split_size)
        #     train_index = get_altset_iloc(sub_pp, test_index)
        #     splits.append((train_index, test_index))

        # # apply GridSearchCV, internally applies folding like defined above for validation
        # clf = GridSearchCV(classifier, parameters, scoring="f1_weighted", cv=splits)

        # clf.fit(x_sub, y_sub)

        suffix = "_binary" if binary else ""
        path = "models/" + name + "_size_" + str(subset_size) + suffix + ".joblib"
        # dump(clf, path)
        clf = load(path)

        # test (not validate)
        precision, recall, f1, y_pred = evaluate_model(clf, f_test, l_test, name + suffix)

        rows.append([len(ix_sub), subset_size, precision, recall, f1])

    return pandas.DataFrame(rows, columns=columns)


debug_ret = None


def load_split():
    both = pd.read_csv("dataset_split/both_preprocessed.csv")
    training = pd.read_csv("dataset_split/training_set.csv")
    testing = pd.read_csv("dataset_split/test_set.csv")

    return both, training, testing


def main():
    simplefilter("ignore")
    global debug_ret
    both, training_df, testing_df = load_split()

    if binary:
        both.loc[both["LABEL"] != "not-ak", "LABEL"] = "ak"
        training_df.loc[training_df["LABEL"] != "not-ak", "LABEL"] = "ak"
        testing_df.loc[testing_df["LABEL"] != "not-ak", "LABEL"] = "ak"

    l_train, l_test = training_df["LABEL"], testing_df["LABEL"]

    increase_step = 100
    kfold_splits = 5

    split = Split(training=training_df["ORIGINAL_INDEX"], testing=testing_df["ORIGINAL_INDEX"])

    vectorizers = [
        Vectorizer(
            name="Tfidf",
            extractor=TfidfVectorizer()
        ),
        Vectorizer(
            name="Count",
            extractor=CountVectorizer()
        ),
    ]
    for vectorizer in vectorizers:
        vectorizer.extract_features(both["CONTENT"])

    classifiers = [
        {"classifier": ComplementNB(), "name": "Complement Naive Bayes", "short_name": "CNB",
         "parameters": {
             "alpha": [0.1],
             "norm": [False]
         }
        },
        {"classifier": DecisionTreeClassifier(), "name": "Decision Tree", "short_name": "DT",
         "parameters": {
            "criterion": ["log_loss"],
            "splitter": ["best"],
            "max_features": [None],
            "class_weight": [None]
         }
        },
        {"classifier": RandomForestClassifier(), "name": "Random Forest", "short_name": "RF",
         "parameters": {
            "n_estimators": [120],
            "criterion": ["log_loss"],
            "max_features": [None],
            "class_weight": ["balanced_subsample"]
         }
        },
        {"classifier": LinearSVC(), "name": "Linear Support Vector Classification", "short_name": "LSV",
         "parameters": {
            "penalty": ["l2"],
            "loss": ["hinge"],
            "dual": [True]
         }
        }
    ]

    # testing_rows = []
    # testing_columns = ["classifier", "vectorizer", "precision", "recall", "f1"]

    evaluations: DataFrame = pd.DataFrame(columns=["classifier",
                                                   "vectorizer",
                                                   "training size",
                                                   "aimed size",
                                                   "precision",
                                                   "recall",
                                                   "f1"])
    for classifier in classifiers:
        logger.info(classifier["name"])

        for vectorizer in vectorizers:
            logger.info(f"\t{vectorizer.name}")
            f_train, f_test = split.split(vectorizer.features)
            start = timeit.default_timer()
            name = classifier["short_name"] + "_" + vectorizer.name
            train_eval = batch_grid_train(f_train, l_train,
                                          f_test, l_test,
                                          classifier["classifier"],
                                          increase_step,
                                          training_df,
                                          classifier["parameters"],
                                          name,
                                          binary)
            stop = timeit.default_timer()

            train_eval["classifier"] = classifier["name"]
            train_eval["vectorizer"] = vectorizer.name
            evaluations = pd.concat([evaluations, train_eval])
            logger.info(f"Done in {stop - start}s")

    evaluations.to_csv(os.path.join("output", "grid_evaluations.csv"))

    # eval_df = pd.DataFrame(testing_rows, columns=testing_columns)
    evaluationsToLatex(evaluations, increase_step, binary)
    # print(eval_df)
    # eval_df.to_csv(os.path.join("output", "testing_on_max.csv"))


def execute_training(classifier, increase_step, kfold_splits, labels_test, labels_train, testing_rows,
                     vectorizer: Vectorizer):
    global debug_ret
    start = timeit.default_timer()
    results, model = batch_train(vectorizer.features, labels_train, classifier["classifier"], increase_step,
                                 kfold_splits)
    stop = timeit.default_timer()

    debug_ret = results
    precision, recall, f1, y_pred = evaluate_model(model, vectorizer.features_test, labels_test)

    testing_rows.append([
        classifier["name"],
        vectorizer.name,
        precision,
        recall,
        f1
    ])

    print_overview(classifier, f1, precision, recall, results, start, stop, vectorizer)

    # This collects the metrics of the latest iteration for each classifier to generate latex bar charts
    classifier[vectorizer.name + "precision"] = results.iloc[-1]["avg_precision"]
    classifier[vectorizer.name + "recall"] = results.iloc[-1][2]  # "avg_recall" for some reason isn't working.
    classifier[vectorizer.name + "f1"] = results.iloc[-1]["avg_f1"]
    printIterationLatex(results, vectorizer, classifier)


def print_overview(classifier, f1, precision, recall, results, start, stop, vectorizer: Vectorizer):
    print(f"""
--------------- {vectorizer.name} -> {classifier["name"]} --- Time: {stop - start} ---------------
{print(results)}
    --- TESTING ---
        precision:  {precision}
        recall:     {recall}
        f1:         {f1}
     ---   ---   ---
---------------------
    """)


if __name__ == '__main__':
    main()
