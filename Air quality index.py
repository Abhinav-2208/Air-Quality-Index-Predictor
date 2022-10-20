#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv("AQI Data.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df=df.dropna()


# In[8]:


df.isnull().sum()


# In[9]:


df.nunique()


# In[10]:


print(df.dtypes)
print("Shape :",df.shape)


# In[11]:


print(df.describe())


# In[12]:


plt.figure(figsize=(9, 8))
sns.distplot(df['PM 2.5'], color='g', bins=100, hist_kws={'alpha': 0.4})


# In[13]:


col= df.columns


# In[14]:


col


# In[15]:


for i in col:
    plt.figure(figsize=(9, 8))
    sns.distplot(df[i], color='g', bins=100, hist_kws={'alpha': 0.4})


# In[16]:


sns.pairplot(df)


# In[17]:


relation= df.corr()


# In[18]:


top_corr_features = relation.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[19]:


df.head()


# In[20]:


import scipy.stats as stat
import pylab


# In[21]:


def plot_curve(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.title(feature)
    plt.show()


# In[22]:


for i in col:
    plot_curve(df,i)


# In[23]:


x=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features


# In[24]:


x.head()


# In[25]:


y.head()


# In[26]:


from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(x,y)


# In[27]:


print(model.feature_importances_)


# In[28]:


feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.plot(kind='barh')
plt.show()


# In[29]:


from sklearn.preprocessing import StandardScaler


# In[30]:


scale= StandardScaler()


# In[31]:


x= scale.fit_transform(x)


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[35]:


regressor.coef_


# In[36]:


regressor.intercept_


# In[37]:


print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))


# In[38]:


print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))


# In[39]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,x,y,cv=5)


# In[40]:


score.mean()


# In[41]:


prediction=regressor.predict(X_test)


# In[42]:


sns.distplot(y_test-prediction)


# In[43]:


plt.scatter(y_test,prediction)


# In[44]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[45]:


from sklearn.metrics import r2_score
print("r2_score is",r2_score(y_test, prediction))


# In[46]:


from xgboost import XGBRegressor


# In[47]:


xg= XGBRegressor()


# In[48]:


from sklearn.model_selection import RandomizedSearchCV


# In[49]:


# No.of Estimators
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
# Different learning rate parameters
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
print(learning_rate)
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
print(max_depth)
#Subssample parameter values
subsample=[0.7,0.6,0.8]
print(subsample)
# Minimum child weight parameters
min_child_weight=[3,4,5,6,7]
print(min_child_weight)


# In[50]:


param= {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}
print(param)


# In[51]:


xgb_random= RandomizedSearchCV(estimator = xg, param_distributions = param,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[ ]:


xgb_random.fit(X_train,y_train)


# In[ ]:


xgb_random.best_params_


# In[ ]:


xgr = XGBRegressor(subsample= 0.8,
 n_estimators= 1100,
 min_child_weight= 3,
 max_depth= 30,
 learning_rate= 0.05)


# In[ ]:


xgr.fit(X_train,y_train)


# In[ ]:


predictions=xgr.predict(X_test)


# In[ ]:


sns.distplot(y_test-predictions)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


from sklearn.metrics import r2_score
print("r2_score is",r2_score(y_test, predictions))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[ ]:


params = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(params)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = params,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[ ]:


rf_random.fit(X_train,y_train)


# In[ ]:


rf_random.best_params_


# In[ ]:


rf1= RandomForestRegressor(n_estimators= 500,
 min_samples_split=  2,
min_samples_leaf=1,
 max_features= 'sqrt',
 max_depth=15)


# In[ ]:


rf1.fit(X_train,y_train)


# In[ ]:


ypred= rf1.predict(X_test)


# In[ ]:


sns.distplot(y_test-ypred)


# In[ ]:


plt.scatter(y_test,ypred)


# In[ ]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, ypred))
print('MSE:', metrics.mean_squared_error(y_test, ypred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ypred)))


# In[ ]:


from sklearn.metrics import r2_score
print("r2_score is",r2_score(y_test, ypred))


# In[ ]:


import pickle


# In[ ]:


# open a file, where you ant to store the data
file = open('aqi_XGBreg_model.pkl', 'wb')

# dump information to that file
pickle.dump(xgr, file)


# In[ ]:


xgr


# In[ ]:


get_ipython().system('pip install tpot')


# In[ ]:


# check tpot version
import tpot
print('tpot: %s' % tpot.__version__)


# In[ ]:


from tpot import TPOTRegressor


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# define evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[ ]:


#define search
model = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)


# In[ ]:


# perform the search
model.fit(X_train, y_train)


# In[ ]:


print(model.score(X_test, y_test))


# In[ ]:


model.fitted_pipeline_


# In[ ]:


pred= model.predict(X_test)


# In[ ]:


sns.distplot(y_test-pred)


# In[ ]:


plt.scatter(y_test,pred)


# In[ ]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[ ]:


from sklearn.metrics import r2_score
print("r2_score is",r2_score(y_test, pred))


# In[ ]:




