from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from preprocess_function import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv('dataset.csv')
# df['comment'] = df['comment'].map(preprocessing)

X_train, X_test, y_train, y_test = train_test_split(df.comment, df.label, test_size=0.2, random_state=42)


steps = []
steps.append(('tfidf', TfidfVectorizer(ngram_range=(1,2),use_idf=True, sublinear_tf = True, norm='l2', smooth_idf=True)))
steps.append(('model',LinearSVC(fit_intercept = True, multi_class='crammer_singer', C=0.2175)))

clf = Pipeline(steps)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#REPORT
report = metrics.classification_report(y_test, y_pred, labels=[1,0], digits=3)
print(report)


# CROSS VALIDATION
cross_score = cross_val_score(clf, X_train,y_train, cv=5)
print("CROSS-VALIDATION 5 FOLDS: %0.3f" % (cross_score.mean()))

# SAVE MODEL
# pickle.dump(clf, open('model.pkl','wb'))

