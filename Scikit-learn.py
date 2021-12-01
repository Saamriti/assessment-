#!/usr/bin/env python
# coding: utf-8

# # Overview of Scikit-learn  Python library

# ### 1.It is simple and efficient tool for predicting data analysis.
# ### 2.It is accessible for everyone using it.
# ### 3.It can be used in various contexts.
# ### 4.It can be build using Numpy,Scipy and matplotlib.
# ### 5.It is an open-source ,commercially usuable.
# ### 6.Scikit-learn is a free machine learning library for Python.

# # Installation

# # Using pip
# ### pip install -U scikit-learn

# # Random forest 

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


dataset = pd.read_csv("petrol_consumption.csv")


# In[5]:


dataset.head()


# In[1]:


import pandas as pd
import plotly.express as px
dataset = pd.read_csv("petrol_consumption.csv")
fig = px.line(dataset,title='Petrol_consumption')
fig.show()


# In[3]:


X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[5]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[6]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[ ]:





# In[11]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[12]:


import numpy as np
import matplotlib.pyplot as plt


# In[28]:


data = {'Mean Absolute Error':51, 'Mean Squared Error':4216, 'Root Mean Squared Error':64}
Metrics = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(Metrics, values, color ='maroon',
        width = 0.4)
plt.xlabel("")
plt.ylabel("")
plt.title("Result")
plt.show()


# In[29]:


import matplotlib.pyplot as plt

x = [51,4216,64]
plt.hist(x, bins = 5)
plt.show()


# # Decision Tree 

# In[16]:


X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[18]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[60]:


from sklearn.tree import DecisionTreeRegressor 

regressor= DecisionTreeRegressor(random_state = 0) 
clf=regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[61]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[3]:


data = {'Mean Absolute Error':50.5, 'Mean Squared Error':4478.7, 'Root Mean Squared Error':66.923090}
Metrics = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(Metrics, values, color ='maroon',
        width = 0.4)
plt.xlabel("")
plt.ylabel("")
plt.title("Result")
plt.show()


# In[16]:


import matplotlib.pyplot as plt

x = [550.5,4478.7,66.923090]
plt.hist(x, bins = 5)
plt.show()


# # Support vector machine(SVM)

# In[22]:


X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[24]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[25]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[26]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[4]:


data = {'Mean Absolute Error':54, 'Mean Squared Error':5819, 'Root Mean Squared Error':76.282899}
Metrics = list(data.keys())
values = list(data.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(Metrics, values, color ='maroon',
        width = 0.4)
plt.xlabel("")
plt.ylabel("")
plt.title("Result")
plt.show()


# In[5]:


import matplotlib.pyplot as plt


# In[15]:


import matplotlib.pyplot as plt

x = [54,5819,76]
plt.hist(x, bins = 5)
plt.show()


# In[ ]:




