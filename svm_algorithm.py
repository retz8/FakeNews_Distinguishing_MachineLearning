import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn import svm #svm algorithm

train_df = pd.read_csv("C:/Users/ekkic/Documents/GitHub/data_anlysis_fake_telos/Train_addcolumn23.csv",encoding = "UTF-8")
test_df = pd.read_csv("C:/Users/ekkic/Documents/GitHub/data_anlysis_fake_telos//Test_addcolumn23.csv",encoding = "UTF-8")

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
# dropna(): column이 비어있다면 없애주고 올리는것
count_vectorizer = CountVectorizer() # word counting
train_vectors = count_vectorizer.fit_transform(train_df["body"])#computer가 읽을  수 있도록 vector로 만듬
test_vectors = count_vectorizer.fit_transform(test_df["body"])
model.fit(train_vectors,train_df["Type"])
preditions = model.predict(test_vectors)
print("Classification Report", classificiation_report (predctions,test_df["Type"])


#clf = svm.SVC(kernel =  "linear")
#scores = model_selection.cross_val_score(clf, train_vectors, train_df["type"], cv=5, scoring = "f1") #cv= cross value 를 몇번 할꺼냐


#print(scores)
#clf.fit(train_vectors, train_df["type"])
#real_scores = clf.score(test_vectors,  test_df["type"])
#print(real_scores)  

