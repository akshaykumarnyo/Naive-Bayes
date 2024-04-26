#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


from sklearn.model_selection import train_test_split


# In[4]:


x,y=load_iris(return_X_y=True)


# In[5]:


x


# In[6]:


y


# In[7]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[8]:


x_train


# In[9]:


x_test


# In[10]:


y_train


# In[11]:


from sklearn.naive_bayes import GaussianNB


# In[14]:


gnb=GaussianNB()


# In[15]:


gnb.fit(x_train,y_train)


# In[16]:


y_pred=gnb.predict(x_test)


# In[17]:


y_pred


# In[22]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[25]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))


# In[26]:


import seaborn as sns


# In[27]:


df=sns.load_dataset("tips")


# In[28]:


df.head()


# In[42]:


df.info()


# In[43]:


df.describe()


# In[29]:


df.isnull().sum()


# In[34]:


df["sex"].unique()


# In[36]:


df["smoker"].unique()


# In[38]:


df["day"].unique()


# In[40]:


df["time"].unique()


# In[48]:


from sklearn.preprocessing import OneHotEncoder


# In[57]:


import pandas as pd

# Initialize OneHotEncoder
onehot_encoder = OneHotEncoder()

# Fit and transform the categorical variable
onehot_encoded = onehot_encoder.fit_transform(df[["sex","smoker","day","time"]])

# Convert the one-hot encoded array to a DataFrame
onehot_encoded_df = pd.DataFrame(onehot_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(["sex","smoker","day","time"]))

# Concatenate the original DataFrame and the one-hot encoded DataFrame
df_encoded = pd.concat([df, onehot_encoded_df], axis=1)

# Display the result
print(df_encoded)


# In[60]:


df_encoded


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




