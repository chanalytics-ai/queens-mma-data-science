#!/usr/bin/env python
# coding: utf-8

# This notebook is the final code that was used for the Kaggle Submission.

# In[2]:


#Import SPARK NLP and NLTK print version
import sparknlp
print("Spark NLP version")
sparknlp.version()

import nltk
nltk.download('stopwords')
from pyspark.sql.functions import col, expr, when
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf

#Import relevant packages for pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from nltk.corpus import stopwords
from pyspark.ml.feature import HashingTF, IDF, CountVectorizer


# In[3]:


# Load in one of the tables (add type field to split out kaggle set after)
df1 = spark.sql("select 'class' as type, * from default.video_games_5")
df2 = spark.sql("select 'class' as type,* from default.books_5_small")
df3 = spark.sql("select 'class' as type,* from default.home_and_kitchen_5_small")
df4 = spark.sql("select 'kaggle' as type,*, 0 as label from default.reviews_kaggle")

#fill nulls from summary field in kaggle set (must keep all rows for submission)
df4 = df4.na.fill('NULLS')

#join base tables
df_complete = df1.union(df2).union(df3)
print((df_complete.count(), len(df_complete.columns)))


# In[4]:


# Drop duplicates
print("Before duplication removal: ", df_complete.count())
#df_distinct = df.dropDuplicates(['reviewerID', 'asin'])
df_distinct = df_complete.dropDuplicates(['reviewerID', 'asin'])
print("After duplication removal: ", df_distinct.count())


# In[5]:


df_help = df_distinct.filter(col("label") == 1)
df_unhelp = df_distinct.filter(col("label") == 0)
print("helpful review: ", df_help.count())
print("unhelpful review: ",df_unhelp.count())


# In[6]:


#randomly sample unhelpful 
df_unhelp_samp = df_unhelp.sample(fraction=0.23, seed=4)
df_full_samp = df_help.union(df_unhelp_samp)


# In[7]:


#Drop from Amazon datazset
drop_list = ['vote', 'reviewTime', 'reviewerID', 'asin', 'reviewerName','unixReviewTime']
df = df_full_samp.select([column for column in df_full_samp.columns if column not in drop_list])

#drop from Kaggle set to match rows/columns
drop_list = ['reviewTime', 'reviewerID', 'asin', 'reviewerName','unixReviewTime']
df4_drop = df4.select([column for column in df4.columns if column not in drop_list])

#join datasets for cleaning
df_combined = df.union(df4_drop)

#set combined as working df
df = df_combined


# In[8]:


#correct datatype of verified feature
df = df.withColumn("verified", when(col("verified") == "true",1)
      .when(col("verified") == "false",0)
      .otherwise("Unknown"))


# In[9]:


from pyspark.sql import functions as sf
#from pyspark.sql.functions import concat_ws,col
df_joined = df.withColumn('joined_column', 
                    sf.concat(sf.col('summary'),sf.lit(' '), sf.col('reviewText')))

#Drop Rows with Null in Joined Column
df_joined=df_joined.filter(sf.col('joined_column').isNotNull())        
print("New number of NAs/Nulls")
newNA=df_joined.select('joined_column').withColumn('isNull_c',sf.col('joined_column').isNull()).where('isNull_c = True').count()
print(newNA)


# In[10]:



#Download stopwords dictionary
eng_stopwords = stopwords.words('english')

#Start SPARK NLP
spark = sparknlp.start()

#Make new Dataframe
#df_spelling=df_remove_unwanted
df_preprocess=df2

#Create different components of the Pipeline steps-> reads in file -> tokenize -> removes stop words -> lemmatizes -> unigrams/ngrams ->outputs columns
document = DocumentAssembler()    .setInputCol("joined_column")    .setOutputCol("document")

tokenizer = Tokenizer()     .setInputCols(["document"])     .setOutputCol("token")     .setSplitChars(['-'])     .setContextChars(['(', ')', '?', '!'])

remover = StopWordsCleaner()    .setInputCols(["token"])    .setOutputCol("nostopwords")    .setStopWords(eng_stopwords)

normalizer = Normalizer()      .setInputCols(['nostopwords'])      .setOutputCol('normalized')      .setLowercase(True)

normalizerCaps = Normalizer()      .setInputCols(['nostopwords'])      .setOutputCol('normalized_wCaps')      .setLowercase(False)

stemmer = Stemmer()     .setInputCols(["normalized"])     .setOutputCol("stem")

lemmatizer = LemmatizerModel.pretrained(name="lemma_antbnc", lang="en")     .setInputCols(["normalized"])     .setOutputCol("lemmat") 

ngram = NGramGenerator()             .setInputCols(["normalized"])             .setOutputCol("ngrams")             .setN(2)             .setEnableCumulative(True)            .setDelimiter("_")
  
pos = PerceptronModel.pretrained('pos_anc')      .setInputCols(['document', 'normalized'])      .setOutputCol('pos')

allowed_tags = ['<JJ>+<NN>', '<NN>+<NN>'] #setting ngrams to meaningful sets for the Chunker function

chunker = Chunker()      .setInputCols(['document', 'pos'])      .setOutputCol('nmeaning')      .setRegexParsers(allowed_tags)

finisher = Finisher()      .setInputCols(['normalized','lemmat','ngrams', 'nmeaning']) #these are the columns that will be printed to the df
  
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


# In[11]:


#NOTE YOU WILL NEED TO SELECT WHICH METHOD YOU WANT for Term Frequency Count and input that column into Line 8
hashingTF2 = HashingTF(inputCol="finished_nmeaning", outputCol="TermFreq_Hash_meaning", numFeatures=2000)
hashingTF = HashingTF(inputCol="finished_ngrams", outputCol="TermFreq_Hash", numFeatures=2000) #For the modelling portion 

#IDF inverse document frequency, is the multipilcation of TF and IDF, results in TF-IDF
idf = IDF(inputCol="TermFreq_Hash", outputCol="TFIDF_features", minDocFreq=10) #SELECT WHICH COLUMN YOU WANT TO PULL FROM 
idf2 = IDF(inputCol="TermFreq_Hash_meaning", outputCol="TFIDF_meaning_features", minDocFreq=10) 

pipeline = Pipeline(stages=[hashingTF, hashingTF2, idf, idf2])

tfidf_model2 = pipeline.fit(df_preprocess)

tfidf_data = tfidf_model2.transform(df_preprocess)


# In[12]:


assembler2 = VectorAssembler(
    inputCols=["TFIDF_features","TFIDF_meaning_features", "overall", "verified"],
    outputCol="features")

output2 = assembler2.transform(data_df)

df_final = output2.select("type","reviewID","features","label")


# In[13]:


df_class = df_final.filter(col("type") == "class")
df_kaggle = df_final.filter(col("type") == "kaggle")


# In[14]:


#remove dummy label and type feature
drop_list = ['label','type']
kaggle = df_kaggle.select([column for column in df_kaggle.columns if column not in drop_list])


# In[15]:


#create working dataset
dataset=df_class

#train-test split
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))


# In[16]:


#regression
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)


# In[17]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
bin_evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', bin_evaluator.evaluate(predictions))


# In[18]:


#CV base reg
Kaggle_predictions_cv = lrModel.transform(kaggle)
sub_reg_cv = Kaggle_predictions_cv.select("reviewID","prediction")


# In[19]:


#submission
display(sub_reg_cv)

