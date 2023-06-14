import os
import re

import jaydebeapi
import nltk
from gensim.models import Word2Vec
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from text_preprocessing import normalize_unicode, expand_contraction, preprocess_text, remove_whitespace

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
    re.compile("On \\d+/\\d+/\\d+ \\d+:\\d+:\\d+ .+ wrote:"),
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
    db_path = os.path.join(os.getcwd(), "../data/dataset/database")
    conn = jaydebeapi.connect("org.h2.Driver",
                              f"jdbc:h2:file:{db_path};IFEXISTS=TRUE;AUTO_SERVER=TRUE",
                              ["", ""],
                              "C:/Program Files (x86)/H2/bin/h2-2.1.214.jar")

    cursor = conn.cursor()
    cursor.execute("SELECT CAST(BODY AS VARCHAR) FROM EMAIL")
    column_data = [row[0] for row in cursor.fetchall()]

    # Pre-process
    input_data = [remove_embedded(email) for email in column_data]

    url_pattern = r'(https?://\S+)'
    input_data = [re.sub(url_pattern, 'url', email) for email in input_data]

    input_data = [email.lower() for email in input_data]

    class_path_pattern = r'.*(org\.|java\.|com\.).*'
    input_data = [re.sub(class_path_pattern, 'class', email) for email in input_data]

    processing_function_list = [
        normalize_unicode,
        expand_contraction]

    input_data = [preprocess_text(email, processing_function_list) for email in input_data]

    special_character_pattern = r'[^a-zA-Z0-9\s]'
    input_data = [re.sub(special_character_pattern, ' ', email) for email in input_data]

    digits_pattern = r'\d+'
    input_data = [re.sub(digits_pattern, '', email) for email in input_data]

    input_data = [remove_whitespace(email) for email in input_data]

    input_data = [remove_stop_words(email) for email in input_data]
    input_data = [lemmatize_sentence(email) for email in input_data]

    input_data = [word_tokenize(email) for email in input_data]

    model = Word2Vec(input_data,
                     vector_size=100,  # Dimensionality of word embeddings
                     workers=4,  # Number of processors (parallelization)
                     window=5,  # Context window for words during training
                     epochs=30)
    sims = model.wv.most_similar('python', topn=10)
    model.save("word2vec_model")
