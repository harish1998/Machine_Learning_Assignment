
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df_por = pd.read_csv("C:\\Users\\HARISH\\Desktop\\MLAssignment\\Student\\student\\student-por.csv",sep=';')


# In[4]:


df_por.head()


# In[5]:


df_por.info()


# In[6]:


def age(age):
    new_age=[]
    for i in age:
        if(i < 17):
            i=0
        elif (i < 19):
            i=1
        else:
            i=2
        new_age.append(i)
    return new_age


# In[7]:


df_por['age']=age(df_por['age'])
sns.boxplot(x='age',y='G3',data=df_por)


# In[8]:


df_por['school']=pd.get_dummies(df_por['school'],drop_first=True)


# In[9]:


sns.boxplot(x='school',y='G3',data=df_por)      # 0 = GP and 1 = MS


# In[10]:


df_por['sex'].value_counts()
sns.boxplot(x='sex',y='G3',data=df_por)


# In[11]:


df_por['address'].value_counts()
sns.boxplot('address','G3',data=df_por)


# In[13]:


df_por['address'] = pd.get_dummies(df_por['address'],drop_first=True)
df_por['famsize'].value_counts()


# In[14]:


sns.boxplot('famsize','G3',data=df_por)


# In[16]:


df_por.drop('famsize',axis=1,inplace=True)


# In[17]:


df_por['Pstatus'].value_counts()
sns.boxplot(x='Pstatus',y='G3',data=df_por)


# In[18]:


df_por.drop('Pstatus',axis=1,inplace=True)


# In[19]:


sns.boxplot(x='Medu',y='G3',data=df_por)


# In[20]:


df_por['Fedu'].value_counts()


# In[21]:


sns.boxplot(x='Fedu',y='G3',data=df_por)


# In[22]:


df_por['Mjob'].value_counts()


# In[23]:


sns.boxplot(x='Mjob',y='G3',data=df_por)


# In[24]:


sns.boxplot(x='Fjob',y='G3',data=df_por)


# In[25]:


df_por['reason'].value_counts()


# In[26]:


sns.boxplot('reason','G3',data=df_por)


# In[27]:


df_por['guardian'].value_counts()


# In[28]:


sns.boxplot(x='guardian',y='G3',data=df_por)


# In[29]:


sns.boxplot(x='traveltime',y='G3',data=df_por)


# In[30]:


df_por.drop('traveltime',axis=1,inplace=True)


# In[31]:


df_por['studytime'].value_counts()


# In[32]:


sns.boxplot(x='studytime',y='G3',data=df_por)


# In[33]:


df_por['failures'].value_counts()


# In[34]:


sns.boxplot(x='failures',y='G3',data=df_por)


# In[35]:


sns.boxplot('schoolsup','G3',data=df_por)


# In[36]:


df_por['schoolsup']=pd.get_dummies(df_por['schoolsup'],drop_first=True)
sns.boxplot('famsup','G3',data=df_por)


# In[37]:


df_por.drop('famsup',axis=1,inplace=True)


# In[38]:


sns.boxplot('paid','G3',data=df_por)


# In[39]:


df_por['paid'].value_counts()


# In[40]:


df_por.drop('paid',axis=1,inplace=True)


# In[41]:


df_por['activities'].value_counts()


# In[42]:


sns.boxplot('activities','G3',data=df_por)


# In[43]:


df_por.drop('activities',axis=1,inplace=True)


# In[44]:


df_por['nursery'].value_counts()


# In[45]:


sns.boxplot('nursery','G3',data=df_por)


# In[46]:


df_por.drop('nursery',axis=1,inplace=True)


# In[47]:


df_por['higher'].value_counts()


# In[48]:


sns.boxplot('higher','G3',data=df_por)


# In[49]:


df_por['higher']=pd.get_dummies(df_por['higher'],drop_first=True)


# In[50]:


df_por['internet'].value_counts()


# In[51]:


df_por['internet']=pd.get_dummies(df_por['internet'],drop_first=True)


# In[52]:


sns.boxplot(x='internet',y='G3',data=df_por)


# In[53]:


df_por['romantic'].value_counts()


# In[54]:


sns.boxplot('romantic','G3',data=df_por)


# In[55]:


df_por.drop('romantic',axis=1,inplace=True)


# In[56]:


df_por['famrel'].value_counts()


# In[57]:


sns.boxplot('famrel','G3',data=df_por)


# In[58]:


df_por.drop('famrel',axis=1,inplace=True)


# In[59]:


df_por['freetime'].value_counts()


# In[60]:


sns.boxplot('freetime','G3',data=df_por)


# In[61]:


df_por.drop('freetime',axis=1,inplace=True)


# In[62]:


df_por['goout'].value_counts()


# In[63]:


sns.boxplot('goout','G3',data=df_por)


# In[64]:


df_por['Dalc'].value_counts()


# In[65]:


sns.boxplot('Dalc','G3',data=df_por)


# In[66]:


sns.boxplot('Walc','G3',data=df_por)


# In[67]:


df_por['health'].value_counts()


# In[68]:


sns.boxplot('health','G3',data=df_por)


# In[69]:


def absences(n):
    new=[]
    for i in n:    
        if (i <= 10):
            i=0
        elif(i <= 20):
            i=1
        else:
            i=2
        new.append(i)
    return new


# In[70]:


df_por['absences']=absences(df_por['absences'])
df_por['absences'].value_counts()                            


# In[72]:


sns.boxplot('absences','G3',data=df_por)


# In[73]:


df_por.info()


# In[74]:


df_por = pd.get_dummies(df_por,drop_first=True)


# In[75]:


df_por.info()


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


def result(score):
    new=[]
    for i in score:
        if (i<8):
            i=0     #Student fails
        else:
            i=1     #student passes
        new.append(i)
    return new


# In[78]:


df_por['G3']=result(df_por['G3'])


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(df_por.drop(['G1','G2','G3'],axis=1), df_por['G3'], test_size=0.33, random_state=42)


# In[81]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.score(X_test,y_test)


# In[94]:


from sklearn import metrics
print(metrics.accuracy_score(y_pred, y_test))
print(metrics.confusion_matrix(y_pred, y_test))
print(metrics.classification_report(y_pred, y_test))

