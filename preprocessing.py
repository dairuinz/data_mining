import pandas as pd
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import gensim as gensim

def vectorizer(work_df):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(work_df['clean'])      #Creates a  sparse matrix (mostly zeros) with words

    vec = pd.DataFrame.sparse.from_spmatrix(bow)        #Puts the vectors in a database
    # print(vec)

    return vec

def wordtovec(work_df):
    # import logging
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # import gensim.downloader as api
    # corpus = api.load('text8')
    # import inspect                                               #downloads ready vocabulary, comment if alreadt saved
    # print(inspect.getsource(corpus.__class__))
    # print(inspect.getfile(corpus.__class__))
    # model = Word2Vec(corpus)
    # model.save('./readyvocab.model')

    # model = Word2Vec(sentences=work_df, vector_size=300, window=5, min_count=1, workers=4, epochs=5)
    model = Word2Vec.load('readyvocab.model')       #reads the vocabulary

    processed_sentences = []
    for sentence in work_df:
        processed_sentences.append(gensim.utils.simple_preprocess(sentence))        #for every sentence in tweets tokenizes each words

    vectors = {}
    i = 0
    for v in processed_sentences:
        vectors[str(i)] = []
        for k in v:
            try:
                vectors[str(i)].append(model.wv[k].mean())      #appends the vector of the word
            except:
                vectors[str(i)].append(np.nan)      #if the word doesnt exist the vocabulary insert it as a Nan value
        i += 1

    df_input = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in vectors.items()]))      #puts the vectors in a dataframe
    df_input.fillna(value=0.0, inplace=True)        #replace Nan values with 0

    df_input = df_input.transpose()     #transposes the matrices in order to insert into the models

    return df_input


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
