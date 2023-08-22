import os
import re

import nltk
import pandas
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text_preprocessing import expand_contraction, normalize_unicode, remove_whitespace
from text_preprocessing import preprocess_text

from dl.constants import TAGS


class PreProcessor:
    def __init__(self, df: pandas.DataFrame):
        """
        This class is responsible for pre-processing the raw email texts. It can then
        be used to save the pre-processed data in JSON format.

        @param df: dataframe containing raw email texts
        """
        self.df = df
        self.relevant_labels = TAGS
        self.lemmatizer = WordNetLemmatizer()
        self.reply_patterns = [
            re.compile('On .+ wrote:'),
            re.compile(".+ hat am \\d+\\.\\d+\\.\\d+ \\d+:\\d+ geschrieben:"),
            re.compile("From:[\\S\\s]*Date:[\\S\\s]*To:[\\S\\s]*Subject:", re.MULTILINE)
        ]

    def remove_embedded(self, raw: str) -> str:
        """
        Remove emails that were embedded in another email using regular expressions.
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

    @staticmethod
    def remove_stop_words(sentence: str) -> str:
        stop_words = set(stopwords.words('english'))
        words = nltk.word_tokenize(sentence)
        filtered_words = [word for word in words if not word.lower() in stop_words]
        return ' '.join(filtered_words)

    def lemmatize_sentence(self, sentence: str) -> str:
        """
        Lemmatization of each word in a sentence.
        """
        words = nltk.word_tokenize(sentence)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def remove_labels(self, labels: list) -> str:
        """
        Removes all labels except for 'existence', 'process', 'property', 'technology', 'not-ak'
        """
        filtered = [label for label in labels if label in self.relevant_labels]
        if 'not-ak' in filtered and len(filtered) > 1:  # This was the case for some reason
            return ''
        return ', '.join(filtered)

    def pre_process(self):
        # Remove irrelevant labels
        self.df['TAGS'] = self.df['TAGS'].str.strip('[]')
        self.df['TAGS'] = self.df['TAGS'].transform(
            lambda x: self.remove_labels(x.split(', '))
        )

        processing_function_list = [normalize_unicode, expand_contraction]
        self.df['BODY'] = self.df['BODY'].transform(
            lambda x: str(x)
        )

        # Normalize unicode and expand contractions
        self.df['BODY'] = self.df['BODY'].transform(
            lambda x: preprocess_text(x, processing_function_list)
        )

        # Parse HTML
        self.df['BODY'] = self.df['BODY'].transform(
            lambda x: BeautifulSoup(x, features="html.parser").get_text()
        )

        # Remove embedded emails
        self.df['BODY'] = self.df['BODY'].transform(
            lambda x: self.remove_embedded(x)
        )

        # To lower case
        self.df['BODY'] = self.df['BODY'].transform(
            lambda x: x.lower()
        )

        # Remove URLs
        url_pattern = r'(https?://\S+)'
        self.df['BODY'] = self.df['BODY'].str.replace(url_pattern, '<url>', regex=True)

        # Replace class paths
        self.df['BODY'] = self.df['BODY'].str.replace(r'\b(org\.|class\.|com\.)[\w.]+', '<classpath>',
                                                      regex=True)

        # Replace special characters
        self.df['BODY'] = self.df['BODY'].str.replace(r'[^a-zA-Z0-9\s]', ' ',
                                                      regex=True)

        # Replace digits
        self.df['BODY'] = self.df['BODY'].str.replace(r'\d+', '', regex=True)

        # Remove stop words
        self.df['BODY'] = self.df['BODY'].transform(
            lambda x: self.remove_stop_words(x)
        )

        # Lemmatization
        self.df['BODY'] = self.df['BODY'].transform(
            lambda x: self.lemmatize_sentence(x)
        )

        # Remove single letters
        self.df['BODY'] = self.df['BODY'].str.replace(r'\b\w\b', '', regex=True)

        # Remove whitespace
        self.df['BODY'] = self.df['BODY'].transform(
            lambda x: remove_whitespace(x)
        )

        # Remove empty emails
        condition = self.df['BODY'] == ''
        self.df = self.df[~condition]

    def export_to_json(self):
        """
        Saves the whole dataset and the labeled part in JSON format
        """
        self.df.to_json(os.path.join(os.getcwd(), 'data/whole_dataset_preprocessed.json'), index=True, orient='index',
                        indent=4)

        # Remove entries that have empty tags
        self.df = self.df[~(self.df['TAGS'] == '')]
        self.df.to_json(os.path.join(os.getcwd(), 'data/labeled_dataset_preprocessed.json'), index=True, orient='index',
                        indent=4)


if __name__ == "__main__":
    df = pd.read_csv('data/dataset_exported.csv')
    pre_processor = PreProcessor(df)
    pre_processor.pre_process()
    pre_processor.export_to_json()
