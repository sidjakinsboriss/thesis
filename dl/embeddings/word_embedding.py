import os
import re

import jaydebeapi
import nltk
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from text_preprocessing import normalize_unicode, expand_contraction, preprocess_text, remove_whitespace

reply_patterns = [
    re.compile('On .+ wrote:'),
    re.compile(".+ hat am \\d+\\.\\d+\\.\\d+ \\d+:\\d+ geschrieben:"),
    re.compile("From:[\\S\\s]*Date:[\\S\\s]*To:[\\S\\s]*Subject:", re.MULTILINE)
]

lemmatizer = WordNetLemmatizer()


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
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


if __name__ == '__main__':
    db_path = os.path.join(os.getcwd(), "../../data/dataset/database")
    conn = jaydebeapi.connect("org.h2.Driver",
                              f"jdbc:h2:file:{db_path};IFEXISTS=TRUE;AUTO_SERVER=TRUE",
                              ["", ""],
                              "C:/Program Files (x86)/H2/bin/h2-2.1.214.jar")

    cursor = conn.cursor()
    cursor.execute("SELECT CAST(BODY AS VARCHAR) FROM EMAIL")
    column_data = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT * FROM EMAIL_TAG")
    data = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]

    # Pre-process
    processing_function_list = [
        normalize_unicode,
        expand_contraction]

    input_data = [preprocess_text(email, processing_function_list) for email in column_data]

    # Parse html
    input_data = [BeautifulSoup(email, features="html.parser").get_text() for email in input_data]

    # Remove embedded emails
    input_data = [remove_embedded(email) for email in input_data]

    # To lower case
    input_data = [email.lower() for email in input_data]

    # Remove urls
    url_pattern = r'(https?://\S+)'
    input_data = [re.sub(url_pattern, '<url>', email) for email in input_data]

    # Remove class paths
    class_path_pattern = r'.*(org\.|java\.|com\.).*'
    input_data = [re.sub(class_path_pattern, '<classpath>', email) for email in input_data]

    # Replace special characters
    special_character_pattern = r'[^a-zA-Z0-9\s]'
    input_data = [re.sub(special_character_pattern, ' ', email) for email in input_data]

    # Replace digits
    digits_pattern = r'\d+'
    input_data = [re.sub(digits_pattern, '', email) for email in input_data]

    # Remove stop words
    input_data = [remove_stop_words(email) for email in input_data]

    # Lemmatization
    input_data = [lemmatize_sentence(email) for email in input_data]

    # Remove single letters
    input_data = [re.sub(r'\b\w\b', '', email) for email in input_data]

    # Remove whitespace
    input_data = [remove_whitespace(email) for email in input_data]

    input_data = [email.split(' ') for email in input_data]

    # with open('whole_dataset_preprocessed.csv', 'w') as file:
    #     for row in input_data:
    #         file.write(','.join(row))
    #         file.write('\n')
    #
    # with open('whole_dataset_preprocessed.csv', 'r') as file:
    #     input_data = [line.strip().split(',') for line in file]

    model = Word2Vec(input_data,
                     vector_size=200,
                     workers=4,
                     min_count=1,
                     sg=1)  # Use skip-gram algorithm
    res = model.wv.most_similar(positive=['java', 'python'])
    model.save('word2vec_model')
