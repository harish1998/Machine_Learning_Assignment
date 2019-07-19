
# coding: utf-8

# In[13]:


import pandas as pd
iris = pd.read_csv("Iris.csv")
iris.drop(['Id'], axis=1, inplace=True)
iris.head()


# In[14]:


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = iris[features]
y = iris.Species


# In[25]:


from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[29]:


from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[30]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[35]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report: \n",metrics.classification_report(y_test, y_pred))
#print("Precision:",metrics.precision_score(y_test, y_pred, average="samples"))
#print("Recall:",metrics.recall_score(y_test, y_pred))

