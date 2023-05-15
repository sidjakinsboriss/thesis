import os
import re

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

reply_patterns = [
    re.compile(
        "On ((Mon|Tue|Wed|Thu|Fri|Sat|Sun), )?"
        "\\d+ (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) \\d+,? "
        "at \\d+:\\d+, .+ wrote:"
    ),
    re.compile(
        "On (Mon|Tue|Wed|Thu|Fri|Sat|Sun), "
        "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec) \\d+, "
        "\\d+,? "
        "at \\d+:\\d+ (AM|PM),? .+ wrote:"
    ),
    re.compile("On \\d+/\\d+/\\d+ \\d+:\\d+, .+ wrote:"),
    re.compile(r"On \\d+/\\d+/\\d+ \\d+:\\d+:\\d+ .+ wrote:"),
    re.compile(".+ hat am \\d+\\.\\d+\\.\\d+ \\d+:\\d+ geschrieben:"),
    re.compile("From:[\\S\\s]*Date:[\\S\\s]*To:[\\S\\s]*Subject:", re.MULTILINE),
]


def remove_embedded(raw: str) -> str:
    """
    Remove emails that were embedded in another.
    In case one is found, the email affected is printed, as well as the portion removed in red.
    :param raw:
    :return:
    """
    # containing indexes where a pattern's first match started
    matches = []
    raw = " ".join(raw.split())

    for pattern in reply_patterns:
        match_found = pattern.search(raw)
        if match_found:
            start_of_chain = match_found.span()[0]
            matches.append(start_of_chain)

    if len(matches) > 0:
        start_of_chain = min(matches)  # get the first match in the text
        stripped = raw[:start_of_chain]
        return stripped

    return raw


def remove_stop_words(sentence: str) -> str:
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(sentence)
    filtered_words = [word for word in words if not word.lower() in stop_words]
    return ' '.join(filtered_words)


def lemmatize_sentence(sentence: str) -> str:
    words = nltk.word_tokenize(sentence)
    lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


def split_dataset(dataframe: pd.DataFrame):
    """
    Split the dataset into 10 subsets, each having the same percentage of architectural/not-ak labels.
    """
    not_ak = dataframe[dataframe["TAGS"] == ["not-ak"]]


def w2v(dataframe: pd.DataFrame):
    model = Word2Vec(sentences=dataframe['CONTENT'].tolist(), window=5, min_count=1, workers=4)
    dataframe['CONTENT'].apply(lambda x: model.wv[x.split()].mean(axis=0))


# TODO: how to deal with tokens that are not in the model
def sentence_vector(sentence: str, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by
    :param raw: raw dataset
    :return: preprocessed dataset
    """
    # get only the columns needed for training
    ret = raw[["SUBJECT", "BODY", "TAGS"]]

    # parse HTML to plain_text and remove embedded threads
    ret["CONTENT"] = ret["BODY"].transform(
        lambda x: remove_embedded(BeautifulSoup(x, features="html.parser").get_text())
    )

    # remove stop words
    ret["CONTENT"] = ret["CONTENT"].transform(
        lambda x: remove_stop_words(x)
    )

    # lemmatization
    ret["CONTENT"] = ret["CONTENT"].transform(
        lambda x: lemmatize_sentence(x)
    )

    # word embedding
    model = Word2Vec(sentences=ret['CONTENT'].tolist(), window=5, min_count=1, workers=4)
    ret["CONTENT"] = ret["CONTENT"].apply(lambda x: sentence_vector(x, model))

    return ret[["CONTENT", "TAGS"]]


if __name__ == "__main__":
    df = pd.read_csv("input.csv")
    pp = preprocess(df)
    pp.to_json(os.path.join(os.getcwd(), "preprocessed.json"), index=True, orient='index', indent=4)
