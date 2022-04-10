#!/usr/bin/env python
# coding: utf-8

# # Case Study 2

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('casestudy.csv')


# In[3]:


df.head()


# In[4]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[26]:


df.info()


# In[27]:


df.describe()


# In[28]:


df.isnull().sum().sort_values(ascending=False)


# ### A) Total Revenue for the Current Year

# In[5]:


df.groupby('year').sum()['net_revenue']


# In[36]:


df.groupby('year').sum()['net_revenue'].plot(kind='bar')


# ### B) New Customer Revenue 

# #### New Customers are those who are present in current year and not in previous year.
# #### So for 2015 we will consider all the customers are new customers because our data starts from it.
# 

# #### For 2016 we will consider customers in 2016 who were not in 2015

# In[6]:


df_1 = df[~df['customer_email'].isin(list(df[df['year'] == 2015]['customer_email']))]
df_1[df_1['year'] == 2016]['net_revenue'].sum()


# #### For 2017 we will consider customers in 2017 who were not in 2016

# In[7]:


df_1 = df[~df['customer_email'].isin(list(df[df['year'] == 2016]['customer_email']))]
df_1[df_1['year'] == 2017]['net_revenue'].sum()


# ### C) Existing Customer Growth

# #### For 2015 the existing customer growth will be equal to the revenue because we dont have the data for 2014

# #### For 2016, we assume existing customers means the customer who still exists from 2015 coming into 2016

# In[8]:


names2016 = set(df[df['year']==2016]['customer_email'])
names2015 = set(df[df['year']==2015]['customer_email'])

cust_exist_2016= list(names2016.intersection(names2015))
cust_exist_2016_df = df[df['customer_email'].isin(cust_exist_2016)]
growth_2016 = cust_exist_2016_df[cust_exist_2016_df["year"]==2016]["net_revenue"].sum() - cust_exist_2016_df[cust_exist_2016_df["year"]==2015]["net_revenue"].sum()

print(growth_2016)


# #### For 2017, we assume existing customers means the customer who still exists from 2016 coming into 2017

# In[9]:


names2017 = set(df[df['year']==2017]['customer_email'])

cust_exist_2017= list(names2017.intersection(names2016))
cust_exist_2017_df = df[df['customer_email'].isin(cust_exist_2017)]
growth_2017 = cust_exist_2017_df[cust_exist_2017_df["year"]==2017]["net_revenue"].sum() - cust_exist_2017_df[cust_exist_2017_df["year"]==2016]["net_revenue"].sum()

print(growth_2017)


# ### D) Revenue lost from attrition

# #### For 2015 the revenue attrition will be 0

# #### 2016

# In[10]:


df_1 = df[~df['customer_email'].isin(list(df[df['year'] == 2016]['customer_email']))]
df_1[df_1['year']==2015]['net_revenue'].sum()


# #### 2017

# In[11]:


df_1 = df[~df['customer_email'].isin(list(df[df['year'] == 2017]['customer_email']))]
df_1[df_1['year']==2016]['net_revenue'].sum()


# ### E) Existing Customer Revenue Current Year

# #### For 2015 the existing customer revenue will be equal to the total 2015 revenue

# #### 2016

# In[12]:


exist_cust_rev_2016 = cust_exist_2016_df[cust_exist_2016_df["year"]==2016]["net_revenue"].sum() 
print(exist_cust_rev_2016)


# #### 2017

# In[13]:


exist_cust_rev_2017 = cust_exist_2017_df[cust_exist_2017_df["year"]==2017]["net_revenue"].sum() 
print(exist_cust_rev_2017)


# ### F) Existing Customer Revenue Prior Year

# #### For 2015 the existing customer revenue will be equal to 0

# #### 2016

# In[14]:


exist_cust_rev_2016_prev = cust_exist_2016_df[cust_exist_2016_df["year"]==2015]["net_revenue"].sum() 
print(exist_cust_rev_2016_prev)


# #### 2017

# In[15]:


exist_cust_rev_2017_prev = cust_exist_2017_df[cust_exist_2017_df["year"]==2016]["net_revenue"].sum() 
print(exist_cust_rev_2017_prev)


# ### G) Total Customers Current Year
# 

# In[16]:


df.groupby('year').count()['customer_email']


# In[32]:


sns.countplot(x=df['year'],data=df)


# ### H) Total Customers Previous Year
# 

# #### In 2015 the total customers previous year will be 0

# #### 2016

# In[17]:


total_prev_2016=len(df[df['year']==2015]['customer_email'].unique())
print(total_prev_2016)


# #### 2017

# In[18]:


l=names2015 and names2016
l=list(set(l))
total_prev_2017=len(l)
print(total_prev_2017)


# ### I) New Customers
# 

# #### In 2015 we have no new customers as we dont have data of 2014

# #### 2016

# In[19]:


new_customers_2016 = (names2016 - ((names2015).intersection(names2016)))
new_customers_2016


# #### 2017

# In[20]:


new_customers_2017 = (names2017 - ((names2016).intersection(names2017)))
new_customers_2017


# ### J) Lost Customers
# 

# #### In 2015 we have no lost customer

# #### 2016

# In[23]:


lost_customers_2016 = (names2015 - ((names2015).intersection(names2016)))
lost_customers_2016


# #### 2017

# In[24]:


lost_customers_2017 = (names2016 - ((names2016).intersection(names2017)))
lost_customers_2017


# ### Visualizing NEW customers

# In[61]:


df_2015 = df[df['year']==2015]
df_2015['NEW'] = 'NEW'


# In[62]:


df_2016 = df[df['year']==2016]

count_df = df[df['year'] != 2017]['customer_email'].value_counts().reset_index()
old = list(count_df[count_df['customer_email']>1]['index'])

df_2016['NEW'] = np.where(df_2016['customer_email'].isin(old), 'OLD', 'NEW')


# In[63]:


df_2017 = df[df['year']==2017]

count_df = df['customer_email'].value_counts().reset_index()
old = list(count_df[count_df['customer_email']>1]['index'])

df_2017['NEW'] = np.where(df_2017['customer_email'].isin(old), 'OLD', 'NEW')


# In[66]:


df_new = pd.concat([df_2015, df_2016, df_2017])


# In[86]:


df_new.groupby(['year', 'NEW']).count().reset_index()


# In[85]:


df_new.groupby(['year', 'NEW']).sum().reset_index()


# In[84]:


sns.countplot(x=df_new['year'],data=df_new,hue=df_new['NEW'])


# #### This plot shows us distribution of new customers over 3 years
