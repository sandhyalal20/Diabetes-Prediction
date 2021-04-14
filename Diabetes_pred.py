#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[4]:


data = pd.read_csv(r'c:\Users\ELCOT\desktop\diabetes.csv.csv')


# In[5]:


data.shape


# In[6]:


data.head(5)


# In[7]:


data.corr


# In[8]:


from sklearn.model_selection import train_test_split

feature_columns = ['Pregnancies','Glucose','SkinThickness','BloodPressure','BMI','Insulin','DiabetesPedigreeFunction','Age']
predicted_class = ['Outcome']

x = data[feature_columns].values
y = data[predicted_class].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state=10)


# In[9]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train,y_train.ravel())

y_pred = gnb.predict(x_test)


# In[10]:


from sklearn import metrics

print("ACCURACY : {0}".format(metrics.accuracy_score(y_test,y_pred)))


# In[11]:


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier = classifier.fit(x_train,y_train)


# In[12]:


y_pred = classifier.predict(x_test)


# In[13]:


from sklearn import metrics


# In[14]:


print("ACCURACY : {0}".format(metrics.accuracy_score(y_test,y_pred)))


# In[15]:


from sklearn import svm


# In[16]:


clf = svm.SVC(kernel = 'linear')
clf.fit(x_train,y_train)


# In[17]:


y_pred = clf.predict(x_test)
print("ACCURACY: {0}".format(metrics.accuracy_score(y_test,y_pred)))


# In[18]:


from sklearn.ensemble import RandomForestClassifier
random_forest_mode1 = RandomForestClassifier(random_state=9)

random_forest_mode1.fit(x_train,y_train.ravel())


# In[19]:


predict_train_data = random_forest_mode1.predict(x_test)

from sklearn import metrics

print("Accuracy: {0}".format(metrics.accuracy_score(y_test,predict_train_data)))


# In[ ]:




