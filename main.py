import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
import preprocessing
import warnings

warnings.filterwarnings('ignore')

def main():
    df = pd.read_csv('amazon.csv')
    work_df = df[:500]

    work_df = preprocessing.data_preprocessing(work_df)

    print(work_df[['Text', 'clean']])

if __name__ == '__main__':
    main()


