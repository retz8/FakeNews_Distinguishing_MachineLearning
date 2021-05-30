import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("C:/Users/ekkic/Desktop/Programming/phyton programming/fakenewsProject/data/Train_addcolumn.csv",encoding = "UTF-8")
test_df = pd.read_csv("C:/Users/ekkic/Desktop/Programming/phyton programming/fakenewsProject/data/Test_addcolumn.csv",encoding = "UTF-8")
test_df["type"] = pd.to_numeric(test_df["type"])
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
# dropna(): column이 비어있다면 없애주고 올리는것
count_vectorizer = feature_extraction.text.CountVectorizer() # word counting
train_vectors = count_vectorizer.fit_transform(train_df["body"])#computer가 읽을  수 있도록 vector로 만듬
test_vectors = count_vectorizer.transform(test_df["body"])

#train_vectors.todense()
model = MultinomialNB()
#model = GaussianNB()

model.fit(train_vectors,train_df["type"])
#scores  = model_selection.cross_val_score(model, train_vectors, train_df["type"], cv=5, scoring = "f1")
#print(scores)
predictions = model.predict(test_vectors)
print("Classification Report", classification_report (predictions,test_df["type"]))
