#!/usr/bin/env python
# coding: utf-8

# # HEALTH INSURANCE CROSS SELL PREDICTION

# # data wrangling

# In[1]:


#read the data
import pandas as pd
import seaborn as sns
import numpy as np

filepath='health-insurance.csv'
data=pd.read_csv(filepath)
data.head()


# In[2]:


data.drop(['id'],axis=1,inplace=True)


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


#finding the missing value
import seaborn as sns
sns.heatmap(data.isnull(),cbar=False,cmap='viridis',yticklabels=False)


# # exploratory data analysis

# In[6]:


#plot
sns.set_style('whitegrid')
sns.countplot(x='Response',data=data)


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x='Vehicle_Age',hue='Response',data=data)


# In[8]:


sns.set_style('whitegrid')
sns.countplot(x='Age',hue='Response',data=data)


# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Gender',hue='Response',data=data)


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='Driving_License',hue='Response',data=data)


# In[11]:


sns.set_style('whitegrid')
sns.countplot(x='Previously_Insured',hue='Response',data=data)


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='Vehicle_Damage',hue='Response',data=data)


# In[13]:


sns.set_style('whitegrid')
sns.catplot(x='Gender',y='Annual_Premium',hue='Response',data=data)


# In[14]:


sns.displot(x='Vintage',hue='Response',data=data)


# In[15]:


data.drop(['Vintage'],axis=1,inplace=True)


# In[16]:


sns.displot(x='Policy_Sales_Channel',hue='Response',data=data)


# In[17]:


data.head(100)


# In[18]:


#convert the object into int type
pd.get_dummies(data['Gender']).head()


# In[19]:


pd.get_dummies(data['Vehicle_Damage']).head()


# In[20]:


pd.get_dummies(data['Vehicle_Age']).head()


# In[21]:


Gender=pd.get_dummies(data['Gender'])


# In[22]:


Vehicle_Age=pd.get_dummies(data['Vehicle_Age'])


# In[23]:


Vehicle_Damage=pd.get_dummies(data['Vehicle_Damage'])


# In[24]:


data=pd.concat([data,Gender,Vehicle_Age,Vehicle_Damage],axis=1)


# In[25]:


data.drop(['Gender','Vehicle_Age','Vehicle_Damage'],axis=1,inplace=True)


# In[26]:


for i in range(len(data['Region_Code'])):
    if(i>=0 and i<10):
        data['Region_Code']=data['Region_Code'].replace({i:0})
    elif(i>=10 and i<20):
        data['Region_Code']=data['Region_Code'].replace({i:1})
    elif(i>=20 and i<30):
        data['Region_Code']=data['Region_Code'].replace({i:2})
    elif(i>=30 and i<40):
        data['Region_Code']=data['Region_Code'].replace({i:3})
    elif(i>=40 and i<=50):
        data['Region_Code']=data['Region_Code'].replace({i:4})


# In[27]:


for i in range(len(data['Policy_Sales_Channel'])):
    if(i>=0 and i<30):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:0})
    elif(i>=30 and i<60):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:1})
    elif(i>=60 and i<120):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:2})
    elif(i>=120 and i<150):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:3})
    elif(i>=150 and i<=160):
        data['Policy_Sales_Channel']=data['Policy_Sales_Channel'].replace({i:4})


# In[28]:


data


# In[29]:


data.corr()


# In[30]:


sns.heatmap(data.corr())


# In[31]:


data.drop('Response',axis=1).head()
data['Response'].head()


# # model prediction

# In[32]:


#Split the data set into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('Response',axis=1),data['Response'], test_size = 0.3)

 


# In[33]:


#Create the model
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()


# In[34]:


#Train the model and create predictions
LR=LR.fit(X_train, y_train)
predictions= LR.predict(X_test)
# predict probabilities
pred_prob= LR.predict_proba(X_test)


# In[35]:


# roc curve for models
from sklearn.metrics import roc_curve
fpr, tpr, thresh = roc_curve(y_test, pred_prob[:,1], pos_label=1)
# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[36]:


#auc score
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_test, pred_prob[:,1])
print(auc_score)


# In[37]:


# plot roc curves
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.plot(fpr, tpr, linestyle='dashed',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='dotted', color='blue')
# title
plt.title('ROC curve')
#  label x and y axis
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show();


# In[38]:


#performance metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[39]:


#Generate a confusion matrix
from sklearn.metrics import confusion_matrix
accutracy=confusion_matrix(y_test,predictions)


# In[40]:


accuracy=confusion_matrix(y_test,predictions)
accuracy


# In[41]:


from sklearn.metrics import accuracy_score


# In[42]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# # Decision tree classifier

# In[43]:


from sklearn.tree import DecisionTreeClassifier


# In[44]:


DTC=DecisionTreeClassifier(criterion='gini',max_features=10,max_depth=5)


# In[45]:


DTC=DTC.fit(X_train,y_train)


# In[46]:


pred=DTC.predict(X_test)


# In[47]:


accuracy=accuracy_score(y_test,pred)
accuracy


# In[48]:


prob=DTC.predict_proba(X_test)


# In[49]:


# roc curve for models
fpr3, tpr3, thresh3 = roc_curve(y_test, prob[:,1], pos_label=1)


# In[50]:


auc_score3= roc_auc_score(y_test, prob[:,1])

print(auc_score3)

