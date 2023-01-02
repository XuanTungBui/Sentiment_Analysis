#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from preprocess_function import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# In[31]:


df = pd.read_csv('dataset.csv')
df.info()


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(df.comment, df.label, test_size=0.2, random_state=42)


# In[57]:


steps = []
steps.append(('tfidf', TfidfVectorizer(ngram_range=(1,2))))
steps.append(('model',MultinomialNB (alpha=1.0, fit_prior=bool, class_prior=None)))

clf = Pipeline(steps)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


report = metrics.classification_report(y_test, y_pred, labels=[1,0], digits=3)
print(report)


# In[58]:


# CROSS VALIDATION
cross_score = cross_val_score(clf, X_train,y_train, cv=5)
print("CROSS-VALIDATION 5 FOLDS: %0.3f" % (cross_score.mean()))


# In[59]:


list_cmt = ['Áo khá mỏng', 'số tiền rẻ phù hợp với hàng','Bọc hàng hơi dở.']
test_list = []
n = len(list_cmt)
for document in list_cmt:
    document = preprocessing(str(document))
    test_list.append(document)
    
pred = clf.predict(test_list)
for i in range(n):
    print(test_list[i]," ",end="")
    print(pred[i])


# In[ ]:




