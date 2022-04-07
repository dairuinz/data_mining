import pandas as pd
import predi
import preprocessing
import warnings

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv('amazon.csv')
    # work_df = df[:15000]
    work_df = df
    # print(work_df.columns)

    work_df = preprocessing.data_preprocessing(work_df)
    # print(work_df[['Text', 'clean']])

    # vectors = preprocessing.vectorizer(work_df)
    # print(vectors.shape)
    vectors = preprocessing.wordtovec(work_df['clean'])
    # print(vectors)

    predi.model_training(vectors, work_df['Score'])

if __name__ == '__main__':
    main()


