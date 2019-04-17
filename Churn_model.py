
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from flask import Flask
import pickle


# In[4]:


raw_data=pd.read_csv("Desktop\user_hack\Telco_Customer_Churn.csv",header=0)
raw_data.head()
raw_data.info()
#dropping unwanted column (column which are not making any sence)
raw_data.drop('customerID',axis=1,inplace=True)
raw_data.drop('Unnamed 32',axis=1,inplace=True)
#checking the data sets that column has been dropped or not 
raw_data.info()
# changing all object type column into numeric type
raw_data.gender=raw_data.gender.map({'Male':1,'Female':0})
raw_data.Partner=raw_data.Partner.map({'Yes':1,'No':0})
raw_data.Dependents=raw_data.Dependents.map({'Yes':1,'No':0})
raw_data.PhoneService=raw_data.PhoneService.map({'Yes':1,'No':0})
raw_data.MultipleLines=raw_data.MultipleLines.map({'Yes':1,'No':0,'No phone service':2})
raw_data.InternetService=raw_data.InternetService.map({'DSL':1,'No':0,'Fiber optic':2})
raw_data.OnlineSecurity=raw_data.OnlineSecurity.map({'Yes':1,'No':0,'No internet service':2})
raw_data.OnlineBackup=raw_data.OnlineBackup.map({'Yes':1,'No':0,'No internet service':2})
raw_data.DeviceProtection=raw_data.DeviceProtection.map({'Yes':1,'No':0,'No internet service':2})
raw_data.TechSupport=raw_data.TechSupport.map({'Yes':1,'No':0,'No internet service':2})
raw_data.StreamingTV=raw_data.StreamingTV.map({'Yes':1,'No':0,'No internet service':2})
raw_data.StreamingMovies=raw_data.StreamingMovies.map({'Yes':1,'No':0,'No internet service':2})
raw_data.Contract=raw_data.Contract.map({'One year':12,'Two year':24,'Month-to-month':1})
raw_data.PaperlessBilling=raw_data.PaperlessBilling.map({'Yes':1,'No':0})
raw_data.PaymentMethod=raw_data.PaymentMethod.map({'Electronic check':1,'Bank transfer (automatic)':2,'Mailed check':3,'Credit card (automatic)':4})
raw_data.Churn=raw_data.Churn.map({'Yes':1,'No':0})

#checking the data sets for all column should have numeric type or not 
raw_data.info()

#replacing empty string with 0 value 
raw_data.replace(' ',0,inplace=True)

#converting data type of TotalCharges column from object to float 
raw_data['TotalCharges']=raw_data.TotalCharges.astype(float)
#checking the informations
raw_data.info()
#checking the sample value of data sets


# In[5]:


filtered_data=raw_data.copy()
highlyrelated_data=filtered_data[['SeniorCitizen','PhoneService','MultipleLines','InternetService','PaperlessBilling','MonthlyCharges','Churn']]


# In[6]:


highlyrelated_data.head()


# In[7]:


corr=highlyrelated_data.corr()
corr


# In[8]:


sns.heatmap(corr)


# In[9]:


sns.heatmap(corr,cmap='coolwarm')


# In[10]:


sns.countplot(highlyrelated_data['Churn'])


# In[11]:


#highlyrelated_data
train,test=train_test_split(highlyrelated_data,test_size=0.3)


# In[12]:


train.shape


# In[13]:


test.shape


# In[14]:


# taking input varaible away and output varaiable away 
#different variables in train set

train.columns


# In[15]:


#Now we want churn as a Y-Axis and rest are the X-Axis
train_x=train.drop('Churn',axis=1)


# In[16]:


# now if we will see churn has been dropped from X axis
train_x.columns


# In[17]:


#assigning churn to y-axis
train_y=train['Churn']


# In[18]:


#will do the same thing for test as well
test_x=test.drop('Churn',axis=1)


# In[19]:


test_y=test['Churn']


# In[20]:


# create the object of the model 
regressor=LogisticRegression()
#fit the model 
regressor.fit(train_x,train_y)


# In[21]:


#PREDICTING THE MODEL ,CALL THE PREDICT FUNCTION FOR MODEL 
pred=regressor.predict(test_x)


# In[22]:


#Accuracy for actual output 
metrics.accuracy_score(pred,test_y)


# In[23]:


#saving model to disk
pickle.dump(regressor,open('churn_model_corr.pkl','wb'))


# In[24]:


#loading model to compare the result
churn_model_corr=pickle.load(open('churn_model_corr.pkl','rb'))


# In[25]:


input_file=pd.read_csv("Desktop\Amfam_hack\predict.csv",header=0)
user_view=input_file.copy()


# In[26]:


input_file.head()


# In[27]:


# changing all object type column into numeric type
input_file.gender=input_file.gender.map({'Male':1,'Female':0})
input_file.Partner=input_file.Partner.map({'Yes':1,'No':0})
input_file.Dependents=input_file.Dependents.map({'Yes':1,'No':0})
input_file.PhoneService=input_file.PhoneService.map({'Yes':1,'No':0})
input_file.MultipleLines=input_file.MultipleLines.map({'Yes':1,'No':0,'No phone service':2})
input_file.InternetService=input_file.InternetService.map({'DSL':1,'No':0,'Fiber optic':2})
input_file.OnlineSecurity=input_file.OnlineSecurity.map({'Yes':1,'No':0,'No internet service':2})
input_file.OnlineBackup=input_file.OnlineBackup.map({'Yes':1,'No':0,'No internet service':2})
input_file.DeviceProtection=input_file.DeviceProtection.map({'Yes':1,'No':0,'No internet service':2})
input_file.TechSupport=input_file.TechSupport.map({'Yes':1,'No':0,'No internet service':2})
input_file.StreamingTV=input_file.StreamingTV.map({'Yes':1,'No':0,'No internet service':2})
input_file.StreamingMovies=input_file.StreamingMovies.map({'Yes':1,'No':0,'No internet service':2})
input_file.Contract=input_file.Contract.map({'One year':12,'Two year':24,'Month-to-month':1})
input_file.PaperlessBilling=input_file.PaperlessBilling.map({'Yes':1,'No':0})
input_file.PaymentMethod=input_file.PaymentMethod.map({'Electronic check':1,'Bank transfer (automatic)':2,'Mailed check':3,'Credit card (automatic)':4})
#converting data type of TotalCharges column from object to float 
raw_data['TotalCharges']=raw_data.TotalCharges.astype(float)


# In[28]:


input_file.head()


# In[29]:


input_file.info()


# In[30]:




feature_set=input_file[['SeniorCitizen','PhoneService','MultipleLines','InternetService','PaperlessBilling','MonthlyCharges']]


# In[31]:


feature_set.head()


# In[32]:


feature_set.info()


# In[33]:


feature_set.columns


# In[34]:


imput_row_list=feature_set.values.tolist()
print(imput_row_list)


# In[35]:


print(len(imput_row_list))


# In[36]:


predicted_putput_list=[]
for i in imput_row_list:
    if((churn_model_corr.predict([i])[0])==1):
        predicted_putput_list.append('Yes')
    else:
        predicted_putput_list.append('No')


# In[37]:


for i in predicted_putput_list:
    print(i)


# In[38]:


user_view['Model_prediction_churn']=predicted_putput_list


# In[39]:


user_view.head()


# In[40]:


final_output=user_view.copy()


# In[41]:


final_output.to_csv("Desktop\user_hack\Model_output\prediction_output_corr.csv",index = False)

