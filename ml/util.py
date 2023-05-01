import logging
import os
from dataclasses import dataclass

import pandas as pd
import random

from time import time
from numpy import floor

from flags import verbose
from formatting import ColorConsoleFormatter
from preprocessing import preprocess

logger = logging.getLogger("Util")
logger.setLevel(logging.DEBUG if verbose else logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColorConsoleFormatter('%(name)s: %(message)s'))
logger.addHandler(handler)


@dataclass
class Thread:
    title: str
    emails = []


def append_or_add(store, key, val):
    if key in store:
        store[key].append(val)
    else:
        store[key] = [val]


def get_threads(subjects: pd.Series):
    """
    Find at which index a thread starts
    :param subjects:
    :return:
    """
    threads = {}

    for iloc, subject in enumerate(subjects.tolist()):
        if subject.startswith("Re:"):
            subject = subject[3:].strip()
        append_or_add(threads, subject, iloc)

    return threads


def get_labels_per_threads(threads, preprocessed):
    raw = {thread: preprocessed.iloc[threads[thread]]["LABEL"].value_counts() for thread in threads.keys()}
    return pd.DataFrame(data=raw).T.fillna(0)


def get_altset_iloc(total, subset_iloc):
    """
    Get the array indexes of the inverse of a subset from a total set
    :param total:
    :param subset_iloc:
    :return:
    """

    subset_iloc_set = set(subset_iloc)
    return [ix for ix in range(total.shape[0]) if ix not in subset_iloc_set]


def contextual_resample_comp(preprocessed: pd.DataFrame, size):
    """
    Shortcut for contextual resample, in case one dataframe holds all data required
    :param preprocessed:
    :param size:
    :return:
    """
    return contextual_resample(preprocessed, preprocessed["SUBJECT"], size)


def contextual_resample(preprocessed: pd.DataFrame, subjects: pd.Series, size):
    """
    Resample in the context of the email dataset: threads will not be split

    :return a set of *iloc* indices from the original dataset, that corresponds to the subset
    """

    # CONTENT, LABEL

    threads = get_threads(subjects)
    labels = preprocessed["LABEL"]

    # list of labels and how many needed, with least occurring label coming first
    optimal_counts = (labels.value_counts() / labels.shape[0] * size).sort_values()

    candidates = []
    thread_count = 0

    labels_per_threads = get_labels_per_threads(threads, preprocessed)
    for label, amount in optimal_counts.iteritems():
        label_counts = labels_per_threads[label]

        # get all threads that would collect a label appropriately
        possible_threads = labels_per_threads[(label_counts > 0) & (label_counts <= amount)]

        # while there are possible threads, and we still need to find labels
        while possible_threads.shape[0] != 0 and labels.iloc[candidates].value_counts().get(label, 0) < amount:
            logger.debug(
                f"current size: {len(candidates)}\n\t{possible_threads.shape[0]} possible threads for getting {amount} of {label}")

            chosen_thread = random.choice(possible_threads.index)
            chosen_emails = threads[chosen_thread]
            candidates.extend(chosen_emails)

            labels_per_threads = labels_per_threads.drop(chosen_thread)
            label_counts = labels_per_threads[label]

            # possible threads is now less, as allowed size has been shortened
            possible_threads = labels_per_threads[(label_counts > 0) & (label_counts <= amount)]

            thread_count += 1

    logger.debug(len(candidates), "in subset")
    logger.debug(thread_count, "threads")
    return candidates


def get_train_test_pair(total, test_ixs):
    return total.iloc[test_ixs], total.drop(test_ixs)


if __name__ == '__main__':
    df = pd.read_csv("input.csv")
    pp = preprocess(df)
    cand = contextual_resample(pp, df["SUBJECT"], floor(df.shape[0] * 0.2))
    subset = pp.iloc[cand]  # ~20%
    altset = pp.drop(subset.index)  # ~80%

    logger.info("total dist")
    logger.info(pp["LABEL"].value_counts() / pp.shape[0])

    logger.info("subset dist")
    logger.info(subset["LABEL"].value_counts() / subset.shape[0])

    orig_rank_name = "ORIGINAL_INDEX"
    folder = f"{int(time())}_division"
    os.mkdir(folder)

    subset.to_csv(os.path.join(folder, "test_set.csv"), index=True, index_label=orig_rank_name)
    altset.to_csv(os.path.join(folder, "training_set.csv"), index=True, index_label=orig_rank_name)
    pp.to_csv(os.path.join(folder, "both_preprocessed.csv"), index=True, index_label=orig_rank_name)
