
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv('C:\\Users\\HARISH\\Desktop\\MLAssignment\\Auto-MPG\\autompg-dataset\\auto-mpg.csv')


# In[4]:


print(data.head())
print(data.index)
print(data.columns)


# In[5]:


data.horsepower.unique()


# In[6]:


data = data[data.horsepower != '?']
print( '?' in data.horsepower )


# In[7]:


data.horsepower = data.horsepower.astype('float')
data.dtypes


# In[8]:


#Add Diesel column as it would affect fuel consumption
data['diesel'] = (data['car name'].str.contains('diesel')).astype(int)


# In[9]:


data.loc[data['diesel'] == 1]


# In[10]:


#Separate Features and Labels

import numpy as np
labels = np.array(data['mpg'])
features = data.drop('mpg', axis=1)


# In[11]:


from sklearn.model_selection import train_test_split

(train_features_f,test_features_f,train_labels,test_labels) = train_test_split(features, 
                                                                               labels, 
                                                                               test_size=0.25, 
                                                                               random_state=4)


# In[12]:


#Remove car name and diesel columns to have only continuous features

train_features_cont = train_features_f.drop(['car name','diesel'], axis=1)
test_features_cont = test_features_f.drop(['car name','diesel'], axis=1)


# In[13]:


#Standarize continuous features and then add binary diesel column

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(train_features_cont)

train_features = np.append(scaler.transform(train_features_cont), 
                           train_features_f['diesel'][:,None], 
                           axis=1)
test_features = np.append(scaler.transform(test_features_cont), 
                          test_features_f['diesel'][:,None], 
                          axis=1)


# In[14]:


print('Shapes')
print('Train features: {0} \nTrain labels: {1}'.format(train_features.shape,
                                                       train_labels.shape))
print('Test features: {0} \nTest labels: {1}'.format(test_features.shape,
                                                     test_labels.shape))


# In[15]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(train_features, train_labels)
test_pred = clf.predict(test_features)


# In[16]:


results = pd.DataFrame({'car name': test_features_f['car name'], 'Test_Labels': test_labels, 'Test_Predict': test_pred})


# In[18]:


print(results.head(5))

