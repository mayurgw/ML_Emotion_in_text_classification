from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
from sklearn import svm


from common import trainTestSplitdf,extractContentSentimentCategoryForSample,preprocessData

if __name__ == '__main__':
	df=extractContentSentimentCategoryForSample('./../../text_emotion.csv')
	df=preprocessData(df)
	max_features = 8000
	X_train, X_test, y_train, y_test = train_test_split(df['content'].values, df['category_id'].values, test_size = 0.10, random_state = 0)
	count_vect = CountVectorizer(stop_words='english',max_features=max_features)

	X_train_counts = count_vect.fit_transform(X_train)
	print(X_train_counts.shape)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	print(X_train_tfidf.shape)
	print(y_train[0:10])
	y_train=np.array(y_train)
	clf_svm = svm.LinearSVC().fit(X_train_tfidf, y_train)


	# count_vect = CountVectorizer()
	X_test_counts = count_vect.transform(X_test)
	X_test_tfidf = tfidf_transformer.transform(X_test_counts)
	print(X_test_tfidf.shape)
	y_pred = clf_svm.predict(X_test_tfidf)
	print(y_pred[0:100])
	print(y_test[0:10])
	print(set(y_test) - set(y_pred))
	print(accuracy_score(y_test, y_pred))
	print(f1_score(y_test, y_pred, average="macro"))
	print(precision_score(y_test, y_pred, average="macro"))
	print(recall_score(y_test, y_pred, average="macro"))
	print(confusion_matrix(y_test,y_pred))