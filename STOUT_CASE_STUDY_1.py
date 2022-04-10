#!/usr/bin/env python
# coding: utf-8

# # CASE STUDY 1

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('loan_1.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum().sort_values(ascending=False)


# ## DATA DESCRIPTION
# ### The given dataset gives us insight about loan interest rates approved to different users and the features that played factor in making that decsion. It has 10000 rows which means it has 10000 different loans that were approved. 
# ### Data has total of 55 features which comparises of different numerical and categorical features. Our target variable is interest rate which is a continuous numerical variable hinting towards a regression problem for our model.
# ### Some features have null value more than 20% of the data which are to be removed and the ones with less than 20% can be filled with some aggregate value.
# ### There are some outliers in the data as well which we will scale up or scale down depending upon their values

# # EDA

# In[6]:


def missing_value_counter(df):
    percent_missing_values = df.isnull().sum()
    percent_missing_percentage = df.isnull().sum() * 100 / len(df)
    
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing_percentage,
                                    'count_missing': percent_missing_values})
    missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)
    return missing_value_df.head(10)


# In[7]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
import missingno
missingno.matrix(df)


# In[8]:


df.drop(['Unnamed: 55','verification_income_joint','annual_income_joint','debt_to_income_joint','months_since_90d_late','months_since_last_delinq','num_accounts_120d_past_due'],axis=1,inplace=True)


# In[9]:


null_cols = ['months_since_last_credit_inquiry', 'emp_length','debt_to_income']


for col in null_cols:
    df[col] = df[col].fillna(
    df[col].dropna().mode().values[0] )   


# In[10]:


df.drop(['state','emp_title','sub_grade'],axis=1,inplace=True)


# In[11]:


plt.figure(figsize=(10,10))
missingno.matrix(df)


# In[12]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr() ,cmap='cubehelix_r')


# In[155]:


num = df.select_dtypes('number').columns.to_list()

#list of all the categoric columns
cat = df.select_dtypes('object').columns.to_list()

#numeric df
loan_num =  df[num]
#categoric df
loan_cat = df[cat]


# # Some PLOTS

# In[144]:


def distplot(data):
    from scipy.stats import norm
    sns.distplot(data['interest_rate'], fit=norm)
    plt.title("Distribution and Skew of Interest Rate")
    plt.xlabel("Interest Rate in %")
    plt.ylabel("Occurance in %")
    plt.show()
    return
distplot(df)


# Most models perform best when the data they deal with is normally distributed (especially linear models). But we see our target variable little right skewed but close enough to normal. The extended tail forward gives a clear sign of a positive skew in our interest rate. This means that there are much more lower values than there are high values. 

# In[145]:


def lines(data):
    sns.lineplot(x=data['emp_length'], y=data['interest_rate'])
    plt.title("Employment Length vs Interest Rate")
    plt.xlabel("Employment Length in yrs")
    plt.ylabel("Interest Rate in %")
    plt.show()
    return
lines(df)


# It seems interest rate vs employment length shows some non-linear relation with a clear drop in average interest rate from working 6 years and 8 years, probably because stability in occupation is a sign of lower risk. The interest rate for people who have worked less than a year seems low though, possibly this is because these are small buisness or enterprise loans that are valued at a lower interest rate so that the buisness itself has a greater chance of success and thus repaying the loan.

# In[147]:


def violin_plot(data):
    sns.violinplot(x="homeownership", y="interest_rate", data=data, hue="term")
    plt.title("Violin Plot")
    plt.xlabel("Home Ownership")
    plt.ylabel("Interest Rate in %")
    plt.show()
    return
violin_plot(df)


# We see that a general trend of increase interest rate with increase in term from 36 months to 60 months. We also see little increase in interest rate within the homeownership as it increases from mortage to rent to own in both the term periods.

# In[154]:


plt.figure(figsize=(15,10))
sns.boxplot(x="loan_purpose", y="interest_rate", data=df)


# The distribution of interest rate on loan purpose is kind of even and this tells customers get loan for various reasons.

# In[159]:


for i in loan_cat.columns: 
    if i != 'loan_status' and i!='loan_purpose':
        
        plt.figure(figsize=(15,10))
        plt.subplot(2,3,1)
        sns.countplot(x=df[str(i)] , data=df ,palette='coolwarm')
        plt.xlabel(i, fontsize=14)


# We see different categorical variables distributed over the data on the count. We see most of the customers option for cash payment. Also the grade of interest rate is covered by A,B,C,D which are lower values. Customers try to provide most details which leads them to get lower interest rate.

# #### Removing Outliers

# In[88]:


def remove_outliers(df, x):
    # Set Limits
    q10, q90 = np.percentile(df[x], 10), np.percentile(df[x], 90)
    iqr = q90 - q10
    cut_off = iqr * 1.5
    lower, upper = q10-cut_off ,  q90 + cut_off
    df[df[x]>upper] = upper
    df[df[x]<lower] = lower
    #print('Outliers of "{}" are removed\n'.format(x))
    return df


# In[89]:


out=loan_num
for item in out.columns:
    outlier_df= remove_outliers(out, item)


# In[90]:


new_df=pd.concat([outlier_df,loan_cat],axis=1)


# In[91]:


new_df


# In[92]:


num = new_df.select_dtypes('number').columns.to_list()

cat = new_df.select_dtypes('object').columns.to_list()

#numeric df
loan_num =  new_df[num]
#categoric df
loan_cat = new_df[cat]


# #### Scaling Data

# In[93]:


q=[]
for i in loan_num.columns:
    q.append(i)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(loan_num),columns=q)


# In[94]:


df2=pd.concat([scaled,loan_cat],axis=1)


# In[95]:


tdf=pd.get_dummies(df2,drop_first=True)


# #### PCA

# In[118]:


def dimensionality_reduction(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=25)
    X = pca.fit_transform(X_train)
    return X
fdf=pd.DataFrame(dimensionality_reduction(tdf))


# In[119]:


np.cumsum(pca.explained_variance_ratio_)


# ### Linear Regression

# In[126]:


X = tdf.drop(columns='interest_rate').iloc[:,:25]
y = tdf['interest_rate']


# In[127]:


from sklearn import metrics
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[134]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state =42)
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[141]:


y_pred = lr.predict(X_test)
y_pred.shape


# In[130]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[167]:


plt.figure(figsize=(10,10))
sns.regplot(x=y_test, y=y_pred)


# RMSE of less than 10% of average is considered good but we have managed to get it under 1%

# ### Ridge Regression

# In[131]:


from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
model = Ridge(alpha=0.51)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)


# In[132]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# RMSE of less than 10% of average is considered good but we have managed to get it under 1%

# ## Assumptions

# #### The data we have has alot of features which increases the dimenionality. After getting dummies for the categorical variables we increase the number of features. We do PCA and select the important 25 features which covers about 95% of the variance in the data. 

# #### The assumtpions made under PCA is that it assumes linearity in data which is present due to regression of the target variables. Another assumption is  that the principal component with high variance must be paid attention and the PCs with lower variance are disregarded as noise. Another assumption about outliers in the dataset is minimum which is taken care of in the EDA.

# #### The linear dependcy data is assumed in forming regression model.

# ## What could I have done better?
# 
# #### The data we have is large but we could have had a larger dataset which had least skewdness and comes from a normal distribution. We know from Central Limit Theorem that if we increase the data enough we approach towards a standard normal. More normalised data will yield better results. I could have cleaned data further and removed extreme values in different features. The number of features were too much which increased the dimenionality. There was room for hyper parameter tuning as paramters are not tuned and could have been set up for better results using techniques like grid search. I could have ensemble methods like bagging/boosting for better results. 
