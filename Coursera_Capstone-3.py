#!/usr/bin/env python
# coding: utf-8

# # Coursera Capstone
# ### This notebook will be used for the applied data science capstone.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


# In[2]:


from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_similarity_score
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn import preprocessing


# In[3]:


print("Hello Capstone Project Course")


# #### New drivers are frequently involved in accidents due to their inexperience. Predicting the severity and type of accidents based on weather and road conditions.

# #### The results of this project will enable driving instructors to adjust their training lessons to better prepare new drivers to deal with these challenges from the start. By training new drivers how to effectively deal with these situations at the beginning of their driving career, the frequency of these types of accidents will decrease preventing injuries and property damage.
# 

# In[4]:


# Download data set  
df_collisions=pd.read_csv('https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv',low_memory=False)
print('Data downloaded and read into a dataframe')


# In[5]:


df_collisions.head()


# In[6]:


# Find out how many entries there are in the dataset
print(df_collisions.shape)
df_collisions.corr()


# In[7]:


# Clean data
df_collisions.columns = list(map(str, df_collisions.columns))
df_collisions['SPEEDING'].replace(to_replace=[np.nan], value=['N'],inplace=True)
df_collisions.dropna(subset = ['ADDRTYPE','WEATHER','ROADCOND','LIGHTCOND'], inplace = True)
print(df_collisions.shape)
df_collisions.head()


# ### Part 1

# #### Visualize data to understand relationships

# In[8]:


df_group1=df_collisions[['WEATHER','LIGHTCOND','SEVERITYCODE']]
group1=df_group1.groupby(['WEATHER','LIGHTCOND'],as_index=False).mean()
group1

group1_pivot=group1.pivot(index='WEATHER',columns='LIGHTCOND')
group1_pivot=group1_pivot.fillna(0)
group1_pivot


# In[9]:


fig, ax = plt.subplots()
im = ax.pcolor(group1_pivot, cmap='RdBu')

#label names
row_labels = group1_pivot.columns.levels[1]
col_labels = group1_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group1_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group1_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[10]:


df_group2=df_collisions[['WEATHER','ADDRTYPE','SEVERITYCODE']]
group2=df_group2.groupby(['WEATHER','ADDRTYPE'],as_index=False).mean()
group2

group2_pivot=group2.pivot(index='WEATHER',columns='ADDRTYPE')
group2_pivot=group2_pivot.fillna(0)
group2_pivot


# In[11]:


fig, ax = plt.subplots()
im = ax.pcolor(group2_pivot, cmap='RdBu')

#label names
row_labels = group2_pivot.columns.levels[1]
col_labels = group2_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group2_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group2_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[12]:


df_group3=df_collisions[['LIGHTCOND','ADDRTYPE','SEVERITYCODE']]
group3=df_group3.groupby(['LIGHTCOND','ADDRTYPE'],as_index=False).mean()
group3

group3_pivot=group3.pivot(index='LIGHTCOND',columns='ADDRTYPE')
group3_pivot=group3_pivot.fillna(0)
group3_pivot


# In[13]:


fig, ax = plt.subplots()
im = ax.pcolor(group3_pivot, cmap='RdBu')

#label names
row_labels = group3_pivot.columns.levels[1]
col_labels = group3_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group3_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group3_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[14]:


df_group4=df_collisions[['WEATHER','SPEEDING','SEVERITYCODE']]
group4=df_group4.groupby(['WEATHER','SPEEDING'],as_index=False).mean()
group4

group4_pivot=group4.pivot(index='WEATHER',columns='SPEEDING')
group4_pivot=group4_pivot.fillna(0)
group4_pivot


# In[15]:


fig, ax = plt.subplots()
im = ax.pcolor(group4_pivot, cmap='RdBu')

#label names
row_labels = group4_pivot.columns.levels[1]
col_labels = group4_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group4_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group4_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[16]:


#weather and road conditions
df_group5=df_collisions[['WEATHER','ROADCOND','SEVERITYCODE']]
group5=df_group5.groupby(['WEATHER','ROADCOND'],as_index=False).mean()
group5

group5_pivot=group5.pivot(index='WEATHER',columns='ROADCOND')
group5_pivot=group5_pivot.fillna(0)
group5_pivot


# In[17]:


fig, ax = plt.subplots()
im = ax.pcolor(group5_pivot, cmap='RdBu')

#label names
row_labels = group5_pivot.columns.levels[1]
col_labels = group5_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group5_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group5_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[18]:


#road and light conditions
df_group6=df_collisions[['ROADCOND','LIGHTCOND','SEVERITYCODE']]
group6=df_group6.groupby(['ROADCOND','LIGHTCOND'],as_index=False).mean()
group6

group6_pivot=group6.pivot(index='ROADCOND',columns='LIGHTCOND')
group6_pivot=group6_pivot.fillna(0)
group6_pivot


# In[ ]:


fig, ax = plt.subplots()
im = ax.pcolor(group6_pivot, cmap='RdBu')

#label names
row_labels = group6_pivot.columns.levels[1]
col_labels = group6_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group6_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group6_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[ ]:


#road and address type
df_group7=df_collisions[['ROADCOND','ADDRTYPE','SEVERITYCODE']]
group7=df_group7.groupby(['ROADCOND','ADDRTYPE'],as_index=False).mean()
group7

group7_pivot=group7.pivot(index='ROADCOND',columns='ADDRTYPE')
group7_pivot=group7_pivot.fillna(0)
group7_pivot


# In[ ]:


fig, ax = plt.subplots()
im = ax.pcolor(group7_pivot, cmap='RdBu')

#label names
row_labels = group7_pivot.columns.levels[1]
col_labels = group7_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group7_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group7_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[ ]:


#road and speeding
df_group8=df_collisions[['ROADCOND','SPEEDING','SEVERITYCODE']]
group8=df_group8.groupby(['ROADCOND','SPEEDING'],as_index=False).mean()
group8

group8_pivot=group8.pivot(index='ROADCOND',columns='SPEEDING')
group8_pivot=group8_pivot.fillna(0)
group8_pivot


# In[ ]:


fig, ax = plt.subplots()
im = ax.pcolor(group8_pivot, cmap='RdBu')

#label names
row_labels = group8_pivot.columns.levels[1]
col_labels = group8_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group8_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group8_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[ ]:


#light and speeding
df_group9=df_collisions[['LIGHTCOND','SPEEDING','SEVERITYCODE']]
group9=df_group9.groupby(['LIGHTCOND','SPEEDING'],as_index=False).mean()
group9

group9_pivot=group9.pivot(index='LIGHTCOND',columns='SPEEDING')
group9_pivot=group9_pivot.fillna(0)
group9_pivot


# In[ ]:


fig, ax = plt.subplots()
im = ax.pcolor(group9_pivot, cmap='RdBu')

#label names
row_labels = group9_pivot.columns.levels[1]
col_labels = group9_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group9_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group9_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[ ]:


#address type and speeding
df_group10=df_collisions[['ADDRTYPE','SPEEDING','SEVERITYCODE']]
group10=df_group10.groupby(['ADDRTYPE','SPEEDING'],as_index=False).mean()
group10

group10_pivot=group10.pivot(index='ADDRTYPE',columns='SPEEDING')
group10_pivot=group10_pivot.fillna(0)
group10_pivot


# In[ ]:


fig, ax = plt.subplots()
im = ax.pcolor(group10_pivot, cmap='RdBu')

#label names
row_labels = group10_pivot.columns.levels[1]
col_labels = group10_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(group10_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group10_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[ ]:


#Create df to analyze types
df_types=df_collisions[['ADDRTYPE','SEVERITYDESC','COLLISIONTYPE','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT','UNDERINFL','WEATHER','ROADCOND','LIGHTCOND','SPEEDING','HITPARKEDCAR']]

for column in df_types.columns:
    print("\n"+column)
    print(df_types[column].value_counts())


# In[ ]:


df_collisiontype=df_collisions['COLLISIONTYPE'].value_counts()
df_collisiontype=pd.DataFrame(df_collisiontype)
df_collisiontype=df_collisiontype.reset_index()
df_collisiontype.columns=['COLLISIONTYPE','COUNT']
df_collisiontype


# In[ ]:


df_weather=df_collisions['WEATHER'].value_counts()
df_weather=pd.DataFrame(df_weather)
df_weather=df_weather.reset_index()
df_weather.columns=['WEATHER','COUNT']
df_weather


# In[ ]:


df_light=df_collisions['LIGHTCOND'].value_counts()
df_light=pd.DataFrame(df_light)
df_light=df_light.reset_index()
df_light.columns=['LIGHTCOND','COUNT']
df_light


# In[ ]:


df_addresstype=df_collisions['ADDRTYPE'].value_counts()
df_addresstype=pd.DataFrame(df_addresstype)
df_addresstype=df_addresstype.reset_index()
df_addresstype.columns=['ADDRTYPE','COUNT']
df_addresstype


# In[ ]:


df_severity=df_collisions['SEVERITYDESC'].value_counts()
df_severity=pd.DataFrame(df_severity)
df_severity=df_severity.reset_index()
df_severity.columns=['SEVERITYDESC','COUNT']
df_severity


# ### Part 2

# #### Determine the variable significance relative to accident severity

# In[ ]:


df_collisions['LIGHTCOND'].replace(to_replace=['Dark - No Street Lights','Dark - Street Lights Off','Dark - Street Lights On','Dark - Unknown Lighting','Dawn','Daylight','Dusk','Other','Unknown'], value=[0,1,2,3,4,5,6,7,8],inplace=True)
df_collisions['ADDRTYPE'].replace(to_replace=['Alley','Block','Intersection'], value=[0,1,2],inplace=True)
df_collisions['ROADCOND'].replace(to_replace=['Dry','Ice','Oil','Other','Sand/Mud/Dirt','Snow/Slush','Standing Water','Unknown','Wet'], value=[0,1,2,3,4,5,6,7,8],inplace=True)
df_collisions['WEATHER'].replace(to_replace=['Blowing Sand/Dirt','Clear','Fog/Smog/Smoke','Other','Overcast','Partly Cloudy','Raining','Severe Crosswind','Sleet/Hail/Freezing Rain','Snowing','Unknown'], value=[0,1,2,3,4,5,6,7,8,9,10],inplace=True)
df_collisions['SPEEDING'].replace(to_replace=['Y','N'], value=[0,1],inplace=True)

Feature=df_collisions[['WEATHER','ROADCOND','LIGHTCOND','ADDRTYPE','SPEEDING']]
Feature.head()

X = Feature
X[0:5]


# In[ ]:


y = df_collisions['SEVERITYCODE'].values
y[0:5]


# In[ ]:


df_collisions[['WEATHER','ROADCOND','LIGHTCOND','ADDRTYPE','SPEEDING','SEVERITYCODE']].corr()


# In[ ]:


pearson_coef, p_value = stats.pearsonr(df_collisions['WEATHER'], df_collisions['SEVERITYCODE'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Since the p-value is  < 0.001, the correlation is statistically significant, and the coefficient shows that the relationship is negative and moderate.


# In[ ]:


pearson_coef, p_value = stats.pearsonr(df_collisions['LIGHTCOND'], df_collisions['SEVERITYCODE'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

#Since the p-value is  <0.001, the correlation is statistically insignificant, and the linear relationship is only moderate.


# In[ ]:


pearson_coef, p_value = stats.pearsonr(df_collisions['ROADCOND'], df_collisions['SEVERITYCODE'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Since the p-value is  < 0.001, the correlation is statistically significant, and the coefficient shows that the relationship is negative and moderate.


# In[ ]:


pearson_coef, p_value = stats.pearsonr(df_collisions['ADDRTYPE'], df_collisions['SEVERITYCODE'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

#Since the p-value is  >0.001, the correlation is not statistically significant, but the linear relationship is very strong


# In[ ]:


pearson_coef, p_value = stats.pearsonr(df_collisions['SPEEDING'], df_collisions['SEVERITYCODE'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# ### Part 3

# #### Evaluate model accuracy

# In[ ]:


# Normalize Data
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[ ]:


#K Nearest Neighbor (KNN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[ ]:


k=2
neigh1=KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat=neigh1.predict(X_test)

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
f10=f1_score(y_test, yhat, average='weighted',labels=np.unique(yhat))
jac0=jaccard_similarity_score(y_test, yhat, sample_weight=None)
print("f1 score: ",f10,"\nJaccard Score: ",jac0)


# In[ ]:


# Decision Tree
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, random_state=4)
print("x: ",X_train1.shape,"y: ",y_train1.shape)
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Tree.fit(X_train1,y_train1)
predTree = Tree.predict(X_test1)

accuracy=metrics.accuracy_score(y_test1, predTree)
print("Decision Tree's Accuracy: ",accuracy)
f11=f1_score(y_test1, predTree, average='weighted',labels=np.unique(predTree))
jac1=jaccard_similarity_score(y_test1, predTree, sample_weight=None)
print("f1 score: ",f11,"\nJaccard Score: ",jac1)


# In[ ]:


# Support Vector Machine
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=4)
print("x: ",X_train2.shape,"y: ",y_train2.shape)

suppvm = svm.SVC(kernel='rbf',gamma='auto')
suppvm.fit(X_train2, y_train2) 
yhat2 = suppvm.predict(X_test2)
f12=f1_score(y_test2, yhat2, average='weighted',labels=np.unique(yhat2))
jac2=jaccard_similarity_score(y_test2, yhat2,sample_weight=None)
print("f1 score: ",f12,"\nJaccard Score: ",jac2)


# In[ ]:


cnf_matrix = confusion_matrix(y_test2, yhat2)
np.set_printoptions(precision=2)

print (classification_report(y_test2, yhat2))


# In[ ]:


# Logistic Regression
X_train3, X_test3, y_train3, y_test3 = train_test_split( X, y, test_size=0.2, random_state=4)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train3,y_train3)
yhat3=LR.predict(X_test3)
yhat3_prob=LR.predict_proba(X_test3)

f13=f1_score(y_test3, yhat3, average='weighted',labels=np.unique(yhat3))
jac3=jaccard_similarity_score(y_test3, yhat3, sample_weight=None)
log3=log_loss(y_test3, yhat3_prob)

print("f1 score: ",f13,"\nJaccard Score: ",jac3,"\nLogLoss: ",log3)


# #### Thank you for reviewing my project!

# In[ ]:




