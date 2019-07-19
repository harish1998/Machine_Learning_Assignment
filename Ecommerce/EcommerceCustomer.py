
# coding: utf-8

# In[24]:


import pandas as pd

#Read the Ecommerce Data
customers = pd.read_csv("Ecommerce Customer.txt", sep=",")
customers.head(3)


# In[25]:


customers.info()


# In[26]:


#Separate Features and Labels
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers[['Yearly Amount Spent']]


# In[27]:


#Do Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)


# In[29]:


X_scaled = scaler.transform(X)
df_X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
df_X_scaled.head()


# In[30]:


#Train Test Split -- Test Size: 30%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_X_scaled, y, test_size=0.3, random_state=101)


# In[31]:


#Build and Train the Linear Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[32]:


predictions = lm.predict(X_test)


# In[35]:


#Visualize Actual Vs Predicted
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[44]:


#Metrics
from sklearn import metrics
import numpy as np
print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test, predictions)))
print('Mean Square Error: {}'.format(metrics.mean_squared_error(y_test, predictions)))
print('Root Mean Square Error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))


# In[48]:


#find Coefficients
coeffs = pd.DataFrame(data=lm.coef_.transpose(), index=X.columns, columns=['Coefficient'])
coeffs

