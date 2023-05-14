import os
import re

import nltk
import pandas as pd
from bs4 import BeautifulSoup
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
    # On 2022/04/06 17:21:52 Dinesh Joshi wrote:

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


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by
    :param raw:
    :return:
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

    return ret[["CONTENT", "TAGS"]]


if __name__ == "__main__":
    df = pd.read_csv("input.csv")
    pp = preprocess(df)
    pp.to_json(os.path.join(os.getcwd(), "preprocessed.json"), index=True, orient='index', indent=4)
