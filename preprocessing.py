import pandas as pd
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def vectorizer(work_df):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(work_df['clean'])      #Creates a  sparse matrix (mostly zeros) with words

    vec = pd.DataFrame.sparse.from_spmatrix(bow)        #Puts the vectors in a database
    # print(vec)

    return vec

def data_preprocessing(work_df):

    work_df['clean'] = work_df['Text'].str.replace('[^a-zA-Z#]', ' ')  # Deletes special characters like :
    work_df['clean'] = work_df['clean'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3]))  # Deletes words with < 3 letters

    text = work_df['clean'].apply(lambda x: x.split())  # Converts words to tokens for easier processing

    stemmer = PorterStemmer()
    text = text.apply(lambda sentence: [stemmer.stem(word) for word in
                                                        sentence])  # Similar words are converted into the shortest similar word

    for i in range(len(text)):
        text.iloc[i] = ' '.join(text.iloc[i])

    work_df['clean'] = text  # Creates new column with the processed text

    return work_df
