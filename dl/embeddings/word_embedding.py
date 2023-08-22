import pandas as pd
from gensim.models import Word2Vec

if __name__ == '__main__':
    "Generates word embeddings by using the whole pre-processed dataset."

    df = pd.read_json('data/whole_dataset_preprocessed.json', orient='index')

    # Choose email texts and split them into words
    emails = df['BODY'].tolist()
    emails = [email.split() for email in emails]

    model = Word2Vec(emails,
                     vector_size=200,
                     workers=4,
                     min_count=1,
                     sg=1)  # Use skip-gram algorithm
    model.save('dl/embeddings/word2vec_model')
