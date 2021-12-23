#!/usr/bin/env python
# coding: utf-8

# This Notebook is the official version of the EDA code which includes the following 
# 
# <li>Loading of the libraries and Amazon Dataframe
# <li>Dataset Statistics
# <li>Pre-Processing 
# <li>Sentiment Analysis
# <li>EDA 
# <li>Export to files (for Vizualisation in Tableau)
# </li>
# 
# Sampling - Cmd 13 requires modification based on the size of the sample that is required for the EDA. Default is 0.40.

# In[2]:


#Import SPARK NLP and NLTK print version
import sparknlp
print("Spark NLP version")
sparknlp.version()

import nltk
nltk.download('stopwords')

#Import relevant packages for pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from nltk.corpus import stopwords


import pyspark.sql.functions as f
from pyspark.sql.functions import length, when


# In[3]:


# Load in one of the tables
df1 = spark.sql("select 'games' as type, * from default.video_games_5")
df2 = spark.sql("select 'books' as type,* from default.books_5_small")
df3 = spark.sql("select 'home' as type,* from default.home_and_kitchen_5_small")

#Add dataset label to determine if its a book, videogame or home review (Nic+Jessee)

df_complete = df1.union(df2).union(df3)
print((df_complete.count(), len(df_complete.columns)))


# ###Dataset Statistics

# In[5]:


#Review per category
df_complete.groupBy('type').count().show()


# In[6]:


#Count of reviews per rating
display(df_complete.groupBy('overall').count())


# In[7]:


#Distinct ReviewerID
from pyspark.sql.functions import countDistinct
df_complete.agg(countDistinct("reviewerID")).show()


# In[8]:


#Number of unique Items
from pyspark.sql.functions import countDistinct
df_complete.agg(countDistinct("asin")).show()


# In[9]:


#Most common review Name
from pyspark.sql.functions import desc
df_complete.groupBy("asin").count().sort(desc("count")).show()


# In[10]:


#Most common review Name
from pyspark.sql.functions import desc
df_complete.groupBy("ReviewerName").count().sort(desc("count")).show()


# In[11]:


#Reviewer with highest count of review
from pyspark.sql.functions import desc
display(df_complete.groupBy("reviewerID").count().sort(desc("count")))


# In[12]:


#Display the reviewer with the highest ammount of reviews.
display(df_complete.filter("ReviewerID = 'A3V6Z4RCDGRC44'").sort(desc("reviewTime")))


# In[13]:


#Creates a sample of the full dataset for the EDA. Recommend using 40%.
df = df_complete.sample(fraction=0.4, seed=4) #this is the dataset without the term frequency 


# In[14]:


# Drop duplicates

df = df.dropDuplicates(['reviewerID', 'asin'])


# In[15]:


# Convert Unix timestamp to readable date

from pyspark.sql.functions import from_unixtime, to_date

df = df.withColumn("reviewTime", to_date(from_unixtime(df.unixReviewTime)))                                                 .drop("unixReviewTime")

# Fill in the empty vote column with 0, and convert it to numeric type

from pyspark.sql.types import *

df = df.withColumn("vote", df.vote.cast(IntegerType()))                                                  .fillna(0, subset=["vote"]) 


# In[16]:


from pyspark.sql import functions as sf
df_joined = df.withColumn('joined_column', 
                    sf.concat(sf.col('summary'),sf.lit(' '), sf.col('reviewText')))

#Drop Rows with Null in Joined Column
df_joined=df_joined.filter(sf.col('joined_column').isNotNull())        
#print("New number of NAs/Nulls")
newNA=df_joined.select('joined_column').withColumn('isNull_c',sf.col('joined_column').isNull()).where('isNull_c = True').count()
#print(newNA)


# In[17]:



#Download stopwords dictionary
eng_stopwords = stopwords.words('english')

#Start SPARK NLP
spark = sparknlp.start()

#Make new Dataframe
df_preprocess=df_joined

#Create different components of the Pipeline steps-> reads in file -> tokenize -> removes stop words -> lemmatizes -> unigrams/ngrams ->outputs columns
document = DocumentAssembler()    .setInputCol("joined_column")    .setOutputCol("document")

tokenizer = Tokenizer()     .setInputCols(["document"])     .setOutputCol("token")     .setSplitChars(['-'])     .setContextChars(['(', ')', '?', '!'])

remover = StopWordsCleaner()     .setInputCols(["token"])    .setOutputCol("nostopwords")    .setStopWords(eng_stopwords)

normalizer = Normalizer()      .setInputCols(['nostopwords'])      .setOutputCol('normalized')      .setLowercase(True)

normalizerCaps = Normalizer()      .setInputCols(['nostopwords'])      .setOutputCol('normalized_wCaps')      .setLowercase(False)

stemmer = Stemmer()     .setInputCols(["normalized"])     .setOutputCol("stem")

lemmatizer = LemmatizerModel.pretrained(name="lemma_antbnc", lang="en")     .setInputCols(["normalized"])     .setOutputCol("lemmat") 

ngram = NGramGenerator()             .setInputCols(["normalized"])             .setOutputCol("ngrams")             .setN(2)             .setEnableCumulative(True)            .setDelimiter("_")
  
pos = PerceptronModel.pretrained('pos_anc')      .setInputCols(['document', 'normalized'])      .setOutputCol('pos')

allowed_tags = ['<JJ>+<NN>', '<NN>+<NN>'] #setting ngrams to meaningful sets for the Chunker function

chunker = Chunker()      .setInputCols(['document', 'pos'])      .setOutputCol('nmeaning')      .setRegexParsers(allowed_tags)

finisher = Finisher()      .setInputCols(['token','nostopwords','normalized','normalized_wCaps','stem','lemmat','ngrams', 'nmeaning']) #these are the columns that will be printed to the df
  
pipeline = Pipeline(
    stages = [
        document,
        tokenizer,
        remover,
        normalizer,
        normalizerCaps,
        stemmer,
        lemmatizer,
        ngram,
        pos,
        chunker,
        finisher
    ])

pipelineModel = pipeline.fit(df_preprocess)
df_preprocess = pipelineModel.transform(df_preprocess)


# In[18]:


import pyspark.sql.functions as f
from pyspark.sql.functions import length, when

#Count number of words
df_preprocess = df_preprocess.withColumn('wordCount', f.size(f.split(f.col('joined_column'), ' ')))
df_preprocess = df_preprocess.withColumn('wordCount_normalized', f.size(f.col('finished_normalized_wCaps')))

#Count number of char
df_preprocess = df_preprocess.withColumn('charCount', length('joined_column'))


# ###Most popular N-Gram over Time

# In[20]:


df_preprocess.cache()
df_word_time = df_preprocess.select(f.explode('finished_nmeaning'), 'reviewTime')


# In[21]:


df_word_time2 = df_word_time.filter(df_word_time.col=="non-stick")
display(df_word_time2)


# In[22]:


df_word_time3 = df_word_time.filter(df_word_time.col=="easy to clean")
display(df_word_time3)


# In[23]:


df_word_time4 = df_word_time.filter(df_word_time.col=="easy to install")
display(df_word_time4)


# In[24]:


df_word_time5 = df_word_time.filter(df_word_time.col=="console game")
display(df_word_time5)


# In[25]:


# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

pydf_to_pd = df_preprocess.select("reviewID", "reviewerID", "asin", "joined_column")
pydf_to_pd
df_pd = pydf_to_pd.toPandas()

#SENTIMENT ANALYSIS
from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df_pd['reviewTextpolarity'] = df_pd['joined_column'].apply(pol)
df_pd['reviewTextsubjectivity'] = df_pd['joined_column'].apply(sub)
#df_small_pd['summarypolarity'] = df_small_pd['summary'].apply(pol)
#df_small_pd['summarysubjectivity'] = df_small_pd['summary'].apply(sub)

#convert pd df to pyspark df
df_pyspark = spark.createDataFrame(df_pd)

#join sentiment using the 3 IDs
df_final = df_preprocess.join(df_pyspark, (df_preprocess.reviewID == df_pyspark.reviewID) &  (df_preprocess.reviewerID == df_pyspark.reviewerID) & (df_preprocess.asin == df_pyspark.asin), how='left')


# In[26]:



df_cleaned = df_final
df_cleaned.createOrReplaceTempView("df_cleaned")


# In[27]:


#Dropping some non required rows to reduce memory requirement.
df_cleaned = df_cleaned.drop('finished_nostopwords','finished_stem', 'finised_lemmat')


# In[28]:


#Number of distinct words per category 
from pyspark.sql.functions import countDistinct
df_cleaned.groupBy('type').agg(countDistinct('finished_normalized_wCaps')).show()


# In[29]:


#Statistics on the number of words (min, max ,avg) per review
df_cleaned.groupBy('type').max("WordCount").show()
df_cleaned.groupBy('type').min("WordCount").show()
df_cleaned.groupBy('type').avg("WordCount").show()


# ###Create Table for tableau

# In[31]:


#data table which includes tokens
word_eda = spark.sql("Select type as Category, overall as Rating, vote as HelpfulVote, verified as Verified, label as Useful, reviewTime as ReviewTime, reviewTextsubjectivity as Subjectivity, finished_ngrams as Tokens_Normalized, reviewTextpolarity as Polarity from df_cleaned")


# In[32]:


word_eda.write.format("parquet").saveAsTable("GMMA2021_new_york.df_eda_word_final")


# In[33]:


#Simplified EDA table without Tokens
eda = spark.sql("Select type as Category, overall as Rating, vote as HelpfulVote, verified as Verified, label as Useful, reviewTime as ReviewTime, reviewTextpolarity as Polarity, reviewTextsubjectivity as Subjectivity, wordCount_normalized as WordCountNormalized, wordCount as WordCount from df_cleaned")


# In[34]:


eda.write.format("parquet").saveAsTable("GMMA2021_new_york.df_eda_final")

