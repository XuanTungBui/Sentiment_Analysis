#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

from preprocess_function import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv('dataset.csv')
df.info()


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(df.comment, df.label, test_size=0.2, random_state=42)


# In[11]:


steps = []
steps.append(('tfidf', TfidfVectorizer(ngram_range=(1,2))))
steps.append(('model', KNeighborsClassifier(n_neighbors = 8, weights='uniform')))

clf = Pipeline(steps)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#REPORT
report = metrics.classification_report(y_test, y_pred, labels=[1,0], digits=3)
print(report)


# In[12]:


# CROSS VALIDATION
cross_score = cross_val_score(clf, X_train,y_train, cv=5)
print("CROSS-VALIDATION 5 FOLDS: %0.3f" % (cross_score.mean()))


# In[13]:


list_cmt = ['Chất lượng số  tuyệt vời','Mong shop khắc phục','Anh shipper nói chuyện ko thân thiện']
test_list = []
for document in list_cmt:
    document = preprocessing(str(document))
    test_list.append(document)
    
pred = clf.predict(test_list)
for i in range(3):
    print(test_list[i]," ",end="")
    print(pred[i])


# In[ ]:




