#!/usr/bin/env python
# coding: utf-8

# Import Necessary Files
# 

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# Load Celebirities Attributes Files.

# In[5]:


attributes=pd.read_csv('archive/list_attr_celeba.csv')

#display first few rows
attributes.head()


# Dataset Overview : A dataframe that includes the image paths and attributes for each image. Ensure that the dataset is correctly loaded and accessible for analysis.

# In[6]:


# Shape of the dataset
print(f"Number of images: {attributes.shape[0]}")
print(f"Number of attributes: {attributes.shape[1]}")

# Check for missing values
print(attributes.isnull().sum())


# In[8]:


# Summary statistics
print(attributes.describe())


# Attribute Distribution

# In[13]:


# Distribution of gender
gender_dist = attributes['Male'].value_counts()
gender_dist.plot(kind='bar', title='Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[14]:


# Distribution of a few other attributes
selected_attributes = ['Smiling', 'Young', 'Wearing_Hat']
attributes[selected_attributes].apply(pd.Series.value_counts).plot(kind='bar', figsize=(12, 6), title='Attribute Distribution')
plt.show()


#  Visualizing Data Distributions

# In[15]:


# Histogram for age Distribution (based on a binary attribute like Young)
attributes['Young'].hist(bins=50)
plt.title('Distribution of Young Attribute')
plt.xlabel('Young (1 = Yes, 0 = No)')
plt.ylabel('Frequency')
plt.show()


# In[16]:


# Histogram for smiling attribute
attributes['Smiling'].hist(bins=50)
plt.title('Distribution of Smiling Attribute')
plt.xlabel('Smiling (1 = Yes, 0 = No)')
plt.ylabel('Frequency')
plt.show()


# In[17]:


# Scatter plot: Gender vs Smiling
sns.scatterplot(x='Male', y='Smiling', data=attributes)
plt.title('Gender vs Smiling')
plt.xlabel('Male (1 = Yes, 0 = No)')
plt.ylabel('Smiling (1 = Yes, 0 = No)')
plt.show()


# In[18]:


# Scatter plot: Young vs Wearing_Hat
sns.scatterplot(x='Young', y='Wearing_Hat', data=attributes)
plt.title('Young vs Wearing Hat')
plt.xlabel('Young (1 = Yes, 0 = No)')
plt.ylabel('Wearing Hat (1 = Yes, 0 = No)')
plt.show()


# Correlation Analysis

# In[9]:


# Correlation matrix
corr_matrix = attributes.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of CelebA Attributes')
plt.show()


# Identifying Outliers

# In[10]:


# Example: Identify outliers in 'Young' attribute using IQR
Q1 = attributes['Young'].quantile(0.25)
Q3 = attributes['Young'].quantile(0.75)
IQR = Q3 - Q1

outliers = attributes[(attributes['Young'] < (Q1 - 1.5 * IQR)) | (attributes['Young'] > (Q3 + 1.5 * IQR))]
print(f"Number of outliers in 'Young' attribute: {outliers.shape[0]}")


# In[11]:


# Visualize outliers in the dataset
sns.boxplot(x='Young', data=attributes)
plt.title('Boxplot of Young Attribute')
plt.show()


# In[ ]:




