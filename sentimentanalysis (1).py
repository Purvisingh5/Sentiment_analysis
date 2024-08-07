#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
df=pd.read_csv("Training_Essay_Data.csv")
df.head()


# In[22]:


df.shape


# In[23]:


df=df.dropna() #removing NaN values
df.shape


# In[24]:


df=df.drop(range(520)) 


# In[26]:


df.shape


# In[70]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
vectorizer1=CountVectorizer(binary=True)
vectorizer2=CountVectorizer(binary=False)
x1=vectorizer1.fit_transform(df.text)
x2=vectorizer2.fit_transform(df.text)
y=df.generated
y.shape


# In[71]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x1,y,test_size=0.25,random_state=41)
model1=BernoulliNB()
model1.fit(xtrain,ytrain)


# In[72]:


x2train,x2test,y2train,y2test=train_test_split(x2,y,test_size=0.25,random_state=41)
model2=MultinomialNB()
model2.fit(x2train,y2train)


# In[73]:


pred=model1.predict(xtest)


# In[77]:


pred2=model2.predict(x2test)


# In[78]:


from sklearn.metrics import accuracy_score , confusion_matrix
a=accuracy_score(ytest,pred)
print("accuracy score for BernoulliNB with Countvectorizer is ",a)
c=confusion_matrix(ytest,pred)
print("confusion matrix for BernoulliNB is \n",c)
m=accuracy_score(y2test,pred2)
print("accuracy score for MultinomialNB with countvectorizer is ",m)
cm=confusion_matrix(y2test,pred2)
print("confusion matrix for MultinomialNB is \n",cm)


# interpret the confusion matrix c: here total 18 datapoints were actually from class 0 but classified to be class 1
# and 79 data points were of class 1 but missclassified to be class 0

# In[33]:


#using MultinomialNB model with Tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2=TfidfVectorizer(stop_words='english') 
x2=vectorizer2.fit_transform(df.text)


# In[34]:


y=df.generated
y.shape
xtrain,xtest,ytrain,ytest=train_test_split(x2,y,test_size=0.25,random_state=30)


# In[35]:


from sklearn.naive_bayes import  MultinomialNB
model=MultinomialNB()
model.fit(xtrain,ytrain)


# In[36]:


pred=model.predict(xtest)


# In[37]:


b=accuracy_score(ytest,pred)
print("accuracy score for MultinomialNB with tfidf is ",b)


# by looking at the accuracy scores of all the three models the highest accuracy score (0.97844)is of model1 i.e  bernoulliNB model using countvectorizer

# In[81]:


#to save the best model for given dataset use joblib
import joblib
joblib.dump(model1,'bernoulli_nb_model.pkl') #model is saved 


# In[82]:


Propermodel = joblib.load('bernoulli_nb_model.pkl')
#reloaded anytime and can be use for predictions

