import json
import os
import re
from collections import Counter

from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word, \
    remove_phone_number, remove_itemized_bullet_and_numbering, expand_contraction, remove_special_character, \
    remove_whitespace, normalize_unicode, remove_stopword, remove_name

import nltk
import numpy as np
import pandas
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class PreProcessor:
    def __init__(self, df: pandas.DataFrame):
        self.df = df
        self.relevant_labels = ['process', 'existence', 'technology', 'property', 'not-ak']
        self.lemmatizer = WordNetLemmatizer()
        self.word_vect = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)
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
            re.compile("On \\d+/\\d+/\\d+ \\d+:\\d+:\\d+ .+ wrote:"),
            re.compile(".+ hat am \\d+\\.\\d+\\.\\d+ \\d+:\\d+ geschrieben:"),
            re.compile("From:[\\S\\s]*Date:[\\S\\s]*To:[\\S\\s]*Subject:", re.MULTILINE)
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
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def sentence_vector(self, sentence: str, model):
        words = nltk.word_tokenize(sentence)
        vectors = [model[word] for word in words if word in model.key_to_index]
        if (length := len(vectors)) <= 300:
            return vectors + [np.zeros(200) for _ in range(300 - length)]
        else:
            return vectors[:300]

    def remove_labels(self, labels: list) -> str:
        filtered = [label for label in labels if label in self.relevant_labels]
        if 'not-ak' in filtered and len(filtered) > 1:  # This was the case for some reason
            return ''
        return ', '.join(filtered)

    def pre_process(self):
        """
        Preprocess the dataset by
        :return: preprocessed dataset
        """
        # get only the columns needed for training
        self.df = self.df[["SUBJECT", "BODY", "TAGS"]]

        # remove unnecessary labels
        self.df['TAGS'] = self.df['TAGS'].str.strip('[]')
        self.df["TAGS"] = self.df["TAGS"].transform(
            lambda x: self.remove_labels(x.split(', '))
        )

        # remove entries that have empty tags
        self.df = self.df[~(self.df['TAGS'] == "")]

        # parse HTML to plain_text and remove embedded threads
        self.df["CONTENT"] = self.df["BODY"].transform(
            lambda x: self.remove_embedded(BeautifulSoup(x, features="html.parser").get_text())
        )

        processing_function_list = [to_lower,
                                    remove_url,
                                    remove_email,
                                    remove_phone_number,
                                    remove_itemized_bullet_and_numbering,
                                    expand_contraction,
                                    remove_special_character,
                                    remove_punctuation,
                                    remove_whitespace,
                                    normalize_unicode,
                                    remove_stopword,
                                    lemmatize_word]

        self.df["CONTENT"] = self.df["CONTENT"].transform(
            lambda x: preprocess_text(x, processing_function_list)
        )

        # # Remove URLs
        # url_pattern = r'(https?://\S+)'
        # self.df['CONTENT'] = self.df['CONTENT'].str.replace(url_pattern, '', regex=True)
        #
        # # remove stop words
        # self.df["CONTENT"] = self.df["CONTENT"].transform(
        #     lambda x: self.remove_stop_words(x)
        # )
        #
        # # lemmatization
        # self.df["CONTENT"] = self.df["CONTENT"].transform(
        #     lambda x: self.lemmatize_sentence(x)
        # )
        #
        # # to lower case
        # self.df["CONTENT"] = self.df["CONTENT"].transform(
        #     lambda x: x.lower()
        # )
        #
        self.df = self.df[["CONTENT", "TAGS"]]

    def word_embedding(self):
        # tokenized_sentences = [word_tokenize(sentence) for sentence in emails.tolist()]
        # model = Word2Vec(sentences=tokenized_sentences, window=5, min_count=2, workers=4, epochs=50)

        self.df["CONTENT"] = self.df["CONTENT"].apply(lambda x: self.sentence_vector(x, self.word_vect))

    def plot_email_word_counts(self):
        length_ranges = [x * 100 for x in range(60)]
        self.df['email_length'] = self.df['CONTENT'].apply(lambda x: len(x))
        self.df['length_range'] = pd.cut(self.df['email_length'], bins=length_ranges)
        email_count = self.df['length_range'].value_counts().sort_index()

        plt.bar(email_count.index.astype(str), email_count.values)
        plt.xlabel('Length Range')
        plt.ylabel('Number of Emails')
        plt.title('Email Count in Length Ranges')
        plt.xticks(rotation=45)
        plt.show()

        self.df = self.df.drop(['email_length', 'length_range'], axis=1)

    def count_word_occurrences(self):
        text = ' '.join(self.df['CONTENT'].values)

        words = text.split()
        word_counts = Counter(words)
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        word_counts_dict = {word: count for word, count in sorted_word_counts}

        # Export the word counts to JSON
        with open('word_counts.json', 'w') as f:
            json.dump(word_counts_dict, f, indent=4)

    def show_words_absent_from_word_embedding(self):
        email_words = list(set(' '.join(self.df['CONTENT'].values).split()))
        model_words = self.word_vect.index_to_key

        with open('words.txt', 'w', encoding="utf-8") as f:
            for word in email_words:
                if word not in model_words:
                    f.write(f"{word}\n")

    def export_to_json(self):
        self.df.to_json(os.path.join(os.getcwd(), "../data/preprocessed.json"), index=True, orient='index', indent=4)


if __name__ == "__main__":
    df = pd.read_csv("../data/data.csv")
    pre_processor = PreProcessor(df)
    pre_processor.pre_process()
    # pre_processor.show_words_absent_from_word_embedding()
    # pre_processor.count_word_occurrences()
    pre_processor.word_embedding()
    pre_processor.export_to_json()
