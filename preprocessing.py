import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
import numpy as np

# def vectorizer(tweets):
#     bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
#     bow = bow_vectorizer.fit_transform(tweets['clean'])      #creates a  sparse matrix (mostly zeros) with words
#
#     vec = pd.DataFrame.sparse.from_spmatrix(bow)        #puts the vectors in a database
#     # print(vec)
#
#     return vec
#     # return bow

def data_preprocessing(work_df):

    work_df['clean'] = work_df['Text'].str.replace('[^a-zA-Z#]', ' ')  # deletes special characters like :
    work_df['clean'] = work_df['clean'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3]))  # deletes words with < 3 letters

    token_tweets = work_df['clean'].apply(lambda x: x.split())  # converts words to tokens for easier processing

    stemmer = PorterStemmer()
    token_tweets = token_tweets.apply(lambda sentence: [stemmer.stem(word) for word in
                                                        sentence])  # similar words are converted into the shortest similar word

    for i in range(len(token_tweets)):
        token_tweets.iloc[i] = ' '.join(token_tweets.iloc[i])

    work_df['clean'] = token_tweets  # creates new column with the processed tweets

    return work_df

def remove_pattern(tweets, pattern):    #deletes what we insert
    r = re.findall(pattern, tweets)
    for word in r:
        tweets = re.sub(word, '', tweets)       #subtracts the given word
    return tweets
