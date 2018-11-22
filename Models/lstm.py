import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras import optimizers
import numpy as np

from common import trainTestSplitdf,extractContentSentimentCategoryForSample,preprocessData


def loadWord2Vec(gloveFilePath):
    WordToVec       = dict()
    with open(gloveFilePath) as embeddings_text: 
        for line in embeddings_text:
            content = line.split(" ")
            word        = content[0] 
            word_vector = np.array(content[1:],dtype='float32')
            WordToVec[word] = word_vector
    return WordToVec

if __name__ == '__main__':
    df=extractContentSentimentCategoryForSample('./../../text_emotion.csv')
    df=preprocessData(df)
    max_features = 8000

    tokenizer = Tokenizer(num_words=max_features,lower=True, split=' ')
    tokenizer.fit_on_texts(df['content'].values)
    # print(tokenizer.document_count)
    X = tokenizer.texts_to_sequences(df['content'].values)
    X = pad_sequences(X)
    Y = pd.get_dummies(df['category_id']).values
    X_train,X_test,Y_train,Y_test=trainTestSplitdf(X,Y,testsize = 0.10)
    
    

    WordToVec=loadWord2Vec('./../../glove.twitter.27B.50d.txt')
    embedding_matrix = np.zeros((max_features, 50))

    for word, index in tokenizer.word_index.items():
        embedding_vector = WordToVec.get(word.lower())
#         print(embedding_vector)
        if embedding_vector is not None and index<max_features:
            embedding_matrix[index] = embedding_vector


    #model 

    embed_dim = 50
    lstm_out = 64

    adam = optimizers.Adam(lr=0.01)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model = Sequential()
    # model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
    model.add(Embedding(max_features, embed_dim, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(50,activation='relu'))
    model.add(Dense(13,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])
    print(model.summary())


    #validation set

    val_size = int(0.5*X_test.shape[0])

    X_validate = X_test[0:val_size]
    Y_validate = Y_test[0:val_size]
    X_test = X_test[val_size:]
    Y_test = Y_test[val_size:]

    # training
    
    batch_size = 32
    history=model.fit(X_train, Y_train, epochs = 500, batch_size=batch_size, verbose = 2,validation_data=(X_validate,Y_validate))

    #testing

    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))