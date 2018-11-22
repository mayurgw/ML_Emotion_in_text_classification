import pandas as pd
from io import StringIO
import re
from nltk.corpus import stopwords
engStopWords = stopwords.words("english")

def trainTestSplitdf(X,Y,testsize=0.2):
    # X = X.sample(frac=1,random_state=random_seed).reset_index(drop=True)
    # Y = Y.sample(frac=1,random_state=random_seed).reset_index(drop=True)
    train_size=X.shape[0]-int(testsize*X.shape[0])
    X_test=X[train_size:]
    X_train=X[0:train_size]
    Y_test=Y[train_size:]
    Y_train=Y[0:train_size]
    
    return X_train,X_test,Y_train,Y_test


def extractContentSentimentCategoryForSample(filepath,sampleSize=-1,random_seed=0):
    # df = pd.read_csv('./../../text_emotion.csv')

    df = pd.read_csv(filepath)
    col = ['sentiment', 'content']
    df = df[col]
    df = df[pd.notnull(df['content'])]
    if(sampleSize!=-1):
        df = df.head(sampleSize)
    df['category_id'] = df['sentiment'].factorize()[0]
    df = df.sample(frac=1,random_state=random_seed).reset_index(drop=True)
    #below code is to segregate category and sentiment
    # category_id_df = df[['sentiment', 'category_id']].drop_duplicates().sort_values('category_id')
    # category_to_id = dict(category_id_df.values)
    # id_to_category = dict(category_id_df[['category_id', 'sentiment']].values)
    return df
    
def preprocessData(df,replace_user_reference='<user>',replace_webpage='<url>'):
    regex_user_reference = re.compile('@[a-zA-Z0-9]+')
    references = []
    df['content'] = df['content'].str.replace(regex_user_reference,replace_user_reference)
    regex_websites = re.compile('http://[www.]*[a-zA-Z0-9]+.[a-z]+/[a-zA-Z0-9//]*')
    websites = []
    df['content'] = df['content'].str.replace(regex_websites,replace_webpage)
    regex_extrachars = re.compile('[\.\:]+')
    df['content'] = df['content'].str.replace(regex_extrachars,'')
    df['content'] = df['content'].str.lower()
    df['content'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in engStopWords]))
    df['content'] = df['content'].str.replace(r"n't", " not ")
    df['content'] = df['content'].str.replace(r"?", " ? ")
    df['content'] = df['content'].str.replace(r"!", " ! ")
    return df


if __name__ == '__main__':
    #sample run
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]], columns=["A", "B"])
    X_train,X_test,Y_train,Y_test=trainTestSplitdf(df['A'],df['B'],.5,1)
    print(X_train.values)
    print(X_test)
    print(Y_train)
    print(Y_test)

    print(extractContentSentimentCategoryForSample('./../../text_emotion.csv',10))