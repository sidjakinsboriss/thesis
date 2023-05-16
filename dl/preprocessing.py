import os
import re

import nltk
import numpy as np
import pandas
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class PreProcessor:
    def __init__(self, df: pandas.DataFrame):
        self.df = df
        self.relevant_labels = ['process', 'existence', 'technology', 'property', 'not-ak']
        self.reply_patterns = [
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

    def remove_embedded(self, raw: str) -> str:
        """
        Remove emails that were embedded in another.
        In case one is found, the email affected is printed, as well as the portion removed in red.
        :param raw:
        :return:
        """
        # containing indexes where a pattern's first match started
        matches = []
        raw = " ".join(raw.split())

        for pattern in self.reply_patterns:
            match_found = pattern.search(raw)
            if match_found:
                start_of_chain = match_found.span()[0]
                matches.append(start_of_chain)

        if len(matches) > 0:
            start_of_chain = min(matches)  # get the first match in the text
            stripped = raw[:start_of_chain]
            return stripped

        return raw

    def remove_stop_words(self, sentence: str) -> str:
        stop_words = set(stopwords.words('english'))
        words = nltk.word_tokenize(sentence)
        filtered_words = [word for word in words if not word.lower() in stop_words]
        return ' '.join(filtered_words)

    def lemmatize_sentence(self, sentence: str) -> str:
        words = nltk.word_tokenize(sentence)
        lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def w2v(self, dataframe: pd.DataFrame):
        model = Word2Vec(sentences=dataframe['CONTENT'].tolist(), window=5, min_count=1, workers=4)
        dataframe['CONTENT'].apply(lambda x: model.wv[x.split()].mean(axis=0))

    # TODO: how to deal with tokens that are not in the model
    def sentence_vector(self, sentence: str, model):
        words = sentence.split()
        vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    def remove_labels(self, labels) -> str:
        labels = [label for label in labels if label in self.relevant_labels]
        return ', '.join(labels)

    def preprocess(self):
        """
        Preprocess the dataset by
        :return: preprocessed dataset
        """
        # get only the columns needed for training
        self.df = self.df[["SUBJECT", "BODY", "TAGS"]]

        # remove unnecessary labels
        self.df['TAGS'] = self.df['TAGS'].str.strip('[]')
        self.df["TAGS"] = self.df["TAGS"].transform(
            lambda x: self.remove_labels(x.split(','))
        )

        # remove entries that have empty tags
        self.df = self.df[~(self.df['TAGS'] == "")]

        # TODO: remove urls

        # parse HTML to plain_text and remove embedded threads
        self.df["CONTENT"] = self.df["BODY"].transform(
            lambda x: self.remove_embedded(BeautifulSoup(x, features="html.parser").get_text())
        )

        # remove stop words
        self.df["CONTENT"] = self.df["CONTENT"].transform(
            lambda x: self.remove_stop_words(x)
        )

        # lemmatization
        self.df["CONTENT"] = self.df["CONTENT"].transform(
            lambda x: self.lemmatize_sentence(x)
        )

        # word embedding
        model = Word2Vec(sentences=self.df['CONTENT'].tolist(), window=5, min_count=1, workers=4)
        self.df["CONTENT"] = self.df["CONTENT"].apply(lambda x: self.sentence_vector(x, model))

        self.df = self.df[["CONTENT", "TAGS"]]

    def export_to_json(self):
        self.df.to_json(os.path.join(os.getcwd(), "../data/preprocessed.json"), index=True, orient='index', indent=4)


if __name__ == "__main__":
    df = pd.read_csv("../data/input.csv")
    pre_processor = PreProcessor(df)
    pre_processor.preprocess()
    pre_processor.export_to_json()
