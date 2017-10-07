
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
df=pd.read_csv('USA_Housing.csv')
df


# In[27]:


fig=sns.pairplot(df)
fig


# In[37]:


fig.savefig("seaborn.jpg")


# In[38]:


fig1=sns.distplot(df['Price'])
figure=fig1.get_figure()
figure.savefig("seaborn1.jpg")


# In[9]:


df.columns


# In[10]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)


# In[11]:


lm.coef_


# In[12]:


X_train.columns


# In[13]:


pd.DataFrame(lm.coef_,X.columns,columns=['Coef'])


# In[17]:


predictions=lm.predict(X_test)
predictions


# In[18]:


y_test


# In[39]:


fig3=sns.distplot(y_test-predictions)
figurepredit=fig3.get_figure()


# In[40]:


figurepredict.savefig('seaborn3.jpg')


# In[49]:


plt.scatter(y_test,predictions)
plt.savefig('predict.jpg')


# In[21]:


from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)


# In[22]:


metrics.mean_squared_error(y_test,predictions)


# In[23]:


np.sqrt(metrics.mean_squared_error(y_test,predictions))


# In[ ]:




