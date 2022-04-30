#!/usr/bin/env python
# coding: utf-8

# # Ted Talks Analysis

# **First, I import the packages I need:**

# In[1]:


import pandas as pd
import matplotlib as plt
import numpy as np


# ### Import the data in DF

# In[2]:


ted_data = pd.read_csv('data.csv')
print(ted_data.head())


# ### I inspect the DF

# In[3]:


print(ted_data.shape)
print(ted_data.dtypes)


# I immediatly note that the date column is not in date format. In any case, I could separate the year in a specific column, in order to compare 2021 vs 2022 (or any other year).

# ### I separate month and year in two new columns

# In[4]:


ted_data['str_split'] = ted_data.date.str.split(' ')
ted_data['month'] = ted_data.str_split.str.get(0)
ted_data['year'] = ted_data.str_split.str.get(1)
ted_data.drop('str_split', axis=1, inplace=True)
print(ted_data.head())


# ## 1. How many videos are created each year?

# In[5]:


ted_data_year = ted_data.groupby('year').title.count().reset_index()
print(ted_data_year)


# When I inspected the first rows of the DF, I never could imagine that there were many videos older than 2 years ago. First videos on TED Talks were created in 1970.
# Interesting thing is that until 2010, the production of videos of TED Talks was not so high and in many years no videos were published.
# This is a table, but how does it appear if I plot this data?

# In[6]:


from matplotlib import pyplot as plt
plt.figure(figsize = (20, 10))
plt.bar(ted_data_year.year, ted_data_year.title, color = 'red')
plt.title('Numbers of Ted Talks during the years', fontsize = 25)
plt.xlabel('Years, from 1970 to 2022', fontsize = 15)
plt.ylabel('Nr. of videos', fontsize = 15)
plt.show()


# ## 2. Like and views distributions

# How are likes and views distributions? Normal distributions? Skewed (left or right)? Let's investigate...

# In[8]:


plt.hist(ted_data.views, bins = 20, color = 'red')
plt.show()


# I immediatly note that most videos have between 0 and 10.000.000 views, but this histogram is not clear, because there are many outliers. So, I create an histogram only taking the more popular portion of dataset.

# In[9]:


plt.hist(ted_data.views, bins = 20, range = (50000, 10000000), color = 'red')
plt.show()


# I've analyzed only videos from 50k to 5 million views. In any case, the distribution is skew-right and is not unimodal, because we can see 3 peaks.

# In[10]:


plt.hist(ted_data.views, bins = 20, range = (50000, 5000000), color = 'red')
plt.show()


# In[11]:


plt.hist(ted_data.likes, bins = 20, color = 'red')
plt.show()


# I notice that we have the same "problem" with likes. Most videos have between 0 and 500.000 likes, but this histogram is not clear, because there are many outliers. So, I create an histogram only taking the more popular portion of dataset.

# In[12]:


plt.hist(ted_data.likes, bins = 20, range = (50000, 500000), color = 'red')
plt.show()


# I've analyzed only videos from 0 to 500k likes. In any case, the distribution is skew-right but is unimodal. In fact, most videos have up to 50.000 likes and are in the first bin of the histogram.

# I can invest if there is a relationship between the number of views and number of likes of a TED Talks videos. Is there?

# In[13]:


import seaborn as sns
sns.scatterplot(data = ted_data, x = 'views', y = 'likes', color = 'red')
plt.xlabel('Number of views')
plt.ylabel('Number of likes')
plt.show()


# It's clear that YES, there is a linear relationship between the two variables. Most views, most likes.
# It could be confirmed with covariance calculations:

# In[14]:


cov_views_likes = np.cov(ted_data.views, ted_data.likes)
print(cov_views_likes)


# In[15]:


from scipy.stats import pearsonr
corr_views_likes, p = pearsonr(ted_data.views, ted_data.likes)
print(corr_views_likes)


# As we can see in both calculations, there is a STRONG linear relationship between the two variables, as already viewed in the scatterplot.

# ## 3. Which is the most popular video?

# In[16]:


best_video_views = ted_data.views.max()

best_video_title = ted_data[ted_data.views == ted_data.views.max()].reset_index()

print('The video with most views has ', best_video_views, ' views and it is: ')
print(best_video_title)


# How many views has the most viewed video each year?

# In[17]:


best_video_per_year = ted_data.groupby('year').views.max().reset_index()
print(best_video_per_year)


# But... which videos are these? I use SQL in Python to investigate...

# In[18]:


import sqldf
query = """
SELECT title, author, year, MAX(views) FROM ted_data
GROUP BY year
ORDER BY MAX(views) desc;
"""
df_view = sqldf.run(query)
print(df_view)


# And, what about the most liked videos for each year? Let's see...

# In[19]:


query = """
SELECT title, author, year, MAX(likes) FROM ted_data
GROUP BY year
ORDER BY MAX(views) desc;
"""
df_like = sqldf.run(query)
print(df_like)


# ### 4. What about the authors? How many authors created videos TED Talks? Are someone created more than 1 video?

# In[20]:


authors = ted_data.groupby('author').title.count().reset_index()
authors = authors.sort_values(by=['title'], ascending=False)
print(authors)


# We can see that there are only 4.443 authors that globally created 5.440 videos.
# Many of them, created only few videos (up to 5) and few of them created more than 5 videos.
# Let's investigate...

# In[21]:


authors_1_5 = authors.loc[authors['title'] < 6]
print(authors_1_5)


# That's the situations: 4.416 authors created at least 1 but no more than 5 videos.

# In[22]:


authors_more_5 = authors.loc[authors['title'] > 5]
print(authors_more_5)


# 27 authors created more than 5 videos. We can notice that the most productive author is Alex Gendler that created 45 videos!
