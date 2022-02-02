#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


comment=pd.read_csv("C:/Users/kumar/Desktop/Data_Science_files/GBcomments.csv",error_bad_lines=False)


# In[3]:


comment.tail()


# In[7]:


pip install textblob


# In[4]:


from textblob import TextBlob


# In[5]:


TextBlob("MEME ME MEME ME MEME ME").sentiment.polarity


# In[6]:


comment.isna().sum()


# In[7]:


comment.dropna(inplace=True)


# comment

# In[8]:


comment


# In[19]:


polarity=[]
for i in comment['comment_text']:
    polarity.append(TextBlob(i).sentiment.polarity)


# In[20]:


comment['polarity']=polarity


# In[21]:


comment


# ## Wordcloud

# In[32]:


pip install wordcloud


# In[22]:


comment_positive=comment[comment['polarity']==1]
comment_positive


# In[23]:


from wordcloud import WordCloud,STOPWORDS


# In[24]:


stopwords=set(STOPWORDS)


# In[25]:


total_comment=" ".join(comment_positive['comment_text'])


# In[26]:


wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comment)


# In[27]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[28]:


comment_nagative=comment[comment['polarity']==-1]
comment_nagative


# In[29]:


total_comment1=" ".join(comment_nagative['comment_text'])


# In[30]:


wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comment1)


# In[31]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# # Analysing Tag columns

# In[32]:


videos=pd.read_csv("C:/Users/kumar/Desktop/Data_Science_files/USvideos.csv",error_bad_lines=False)


# In[33]:


videos.head()


# In[34]:


tags_complete=" ".join(videos['tags'])


# In[35]:


import re


# In[36]:


tags=re.sub('[^a-zA-Z]'," ",tags_complete)


# In[37]:


tags


# In[38]:


re.sub(' +'," ",tags_complete)


# In[39]:


wordcloud=WordCloud(width=1000,height=500,stopwords=set(STOPWORDS)).generate(tags)


# In[40]:


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# # analysing like, dislike and views and relation between them

# In[41]:


sns.regplot(data=videos,x='views',y='dislikes')
plt.title('regression relation between views and dislikes')


# In[42]:


sns.regplot(data=videos,x="views",y='likes')
plt.title("regression relation between views and likes ")


# In[43]:


df_corr=videos[['views','likes','dislikes']]
df_corr.corr()


# In[44]:


sns.heatmap(df_corr.corr(),annot=True)


# # Analysing emojis in comment

# In[52]:


pip install emoji


# In[45]:


import emoji


# In[46]:


comment1=comment['comment_text'][1]


# In[47]:


[c for c in comment1 if c in emoji.UNICODE_EMOJI_ENGLISH]


# In[51]:


str=''
for i in comment['comment_text']:
   list = [c for c in i if c in emoji.UNICODE_EMOJI_ENGLISH]
   for ele in list:
        str=str+ele
    


# In[52]:


len(str)


# In[53]:


str


# In[54]:


result={}
for i in set(str):
    result[i]=str.count(i)


# In[55]:


result


# In[57]:


final={}
for key,value in sorted(result.items(),key=lambda item:item[1]):
    final[key]=value


# In[58]:


final


# In[60]:


keys=[*final.keys()]


# In[62]:


values=[*final.values()]


# In[65]:


df=pd.DataFrame({'chars':keys[-20:],'num':values[-20: ]})
df


# In[66]:


pip install plotly


# In[67]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[69]:


trace=go.Bar(
x=df['chars'],
y=df['num']
)
iplot([trace])


# In[ ]:





# In[ ]:




