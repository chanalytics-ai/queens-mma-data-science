#!/usr/bin/env python
# coding: utf-8

# This notebook is the final code for the Topic Model and the associated words. At the default configuration, the sample size is 0.05. It is recommended to augment to 0.4 to reproduce the results showed in the presentation.

# In[2]:


#Import SPARK NLP and NLTK print version
import sparknlp
print("Spark NLP version")
sparknlp.version()

import nltk
nltk.download('stopwords')

import pyspark.sql.functions as F

#Import relevant packages for pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from nltk.corpus import stopwords


# In[3]:


# Load in one of the tables
df1 = spark.sql("select 'games' as type, * from default.video_games_5")
df2 = spark.sql("select 'books' as type,* from default.books_5_small")
df3 = spark.sql("select 'home' as type,* from default.home_and_kitchen_5_small")

#Add dataset label to determine if its a book, videogame or home review (Nic+Jessee)

df = df1.union(df2).union(df3)
print((df.count(), len(df.columns)))


# In[4]:


#Must change fraction value for sample size required. Default =0.05 simply to minimize compute time. Recommended meaningful sample size is 0.4.
df = df.sample(fraction=0.05, seed=4) 


# In[5]:


#display the small dataset
display(df)


# In[6]:


# Drop duplicates - only if you use the small/medium dataset
print("Before duplication removal: ", df.count())
df_distinct = df.dropDuplicates(['reviewerID', 'asin'])
print("After duplication removal: ", df_distinct.count())


# In[7]:


# Convert Unix timestamp to readable date

from pyspark.sql.functions import from_unixtime, to_date

df_with_date = df_distinct.withColumn("reviewTime", to_date(from_unixtime(df_distinct.unixReviewTime)))                                                 .drop("unixReviewTime")

# Fill in the empty vote column with 0, and convert it to numeric type

from pyspark.sql.types import *

df_fill_vote = df_with_date.withColumn("vote", df_with_date.vote.cast(IntegerType()))                                                  .fillna(0, subset=["vote"]) 


# In[8]:


from pyspark.sql import functions as sf
df_joined = df_fill_vote.withColumn('joined_column', 
                    sf.concat(sf.col('summary'),sf.lit(' '), sf.col('reviewText')))

#Drop Rows with Null in Joined Column
df_joined=df_joined.filter(sf.col('joined_column').isNotNull())        
print("New number of NAs/Nulls")
newNA=df_joined.select('joined_column').withColumn('isNull_c',sf.col('joined_column').isNull()).where('isNull_c = True').count()
print(newNA)


# In[9]:


## Extensive custom stopwords list to be removed from the reviews

eng_stopwords_custom = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]


# In[10]:



#Download stopwords dictionary
eng_stopwords = stopwords.words('english')

#Start SPARK NLP
spark = sparknlp.start()

#Make new Dataframe
#df_spelling=df_remove_unwanted
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


# In[11]:


from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from pyspark.sql.functions import udf

import pyspark.sql.functions as f
from pyspark.sql.functions import length, when

#Creating User Defined Functions for Each of the Stemming Options
def pstemming(col):
  p_stemmer = PorterStemmer()
  return [p_stemmer.stem(w) for w in col]

def snowball(col):
  sb_stemmer = SnowballStemmer('english')
  return [sb_stemmer.stem(w) for w in col]

def lancaster(col):
  l_stemmer = LancasterStemmer()
  return [l_stemmer.stem(w) for w in col]

pstemming_udf = udf(pstemming, ArrayType(StringType()))
sbstemming_udf = udf(snowball, ArrayType(StringType()))
lstemming_udf = udf(lancaster, ArrayType(StringType()))

#Running Each of the UDFs to Create new columns for each of the stemming options
df_preprocess = df_preprocess.withColumn("finished_stem_Porter", pstemming_udf(df_preprocess.finished_normalized))

df_preprocess = df_preprocess.withColumn("finished_stem_Snowball", sbstemming_udf(df_preprocess.finished_normalized))

df_preprocess = df_preprocess.withColumn("finished_stem_Lancaster", lstemming_udf(df_preprocess.finished_normalized))

#Count number of words
df_preprocess = df_preprocess.withColumn('wordCount', f.size(f.split(f.col('joined_column'), ' ')))

#Count number of char
df_preprocess = df_preprocess.withColumn('charCount', length('joined_column'))


# In[12]:


from pyspark.ml.feature import HashingTF, IDF, CountVectorizer

vectorizer= CountVectorizer(inputCol='finished_nmeaning', outputCol='TermFreq_CV') 

idf = IDF(inputCol="TermFreq_CV", outputCol="IDF_features", minDocFreq=10) #SELECT WHICH COLUMN YOU WANT TO PULL FROM 

pipeline = Pipeline(stages=[vectorizer, idf]) #took out the hashingTF because running vectorizer, but if you use the vectorizer, put in vectorizer, idf

tfidf_model = pipeline.fit(df_preprocess)

tfidf_data = tfidf_model.transform(df_preprocess)


# In[13]:


#Building the Query result for the Word Count.
counts = df_preprocess.select(f.explode('finished_nmeaning').alias('finished_nmeaning')).groupBy('finished_nmeaning').count().collect()
display(counts)


# In[14]:



from pyspark.ml.clustering import LDA, LDAModel

#can change the number of topics and number of iterations to adjust run speed
number_topics = 50 
max_iteration = 4 

lda = LDA(featuresCol="IDF_features", k= number_topics, maxIter = max_iteration, seed=59, optimizer="online")
ldaModel=lda.fit(tfidf_data)  

#the loglikihood - the higher the better
ll = ldaModel.logLikelihood(tfidf_data) 
#the perplexity - the lower the better
lp = ldaModel.logPerplexity(tfidf_data) 
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))


# Performs the result
lda_df = ldaModel.transform(tfidf_data)


# In[15]:


#shows the result of the LDA - column is topicDistribution
display(lda_df)


# In[16]:


#phi: results from LDA - topic, termIndices, termWeights - phi: which words in which topics

#the maximum terms you want to set per topic needs to be an integer, by order of decreasing weight
maxTerms = 10

#topics described by their top weighted terms
topics = ldaModel.describeTopics(maxTermsPerTopic = maxTerms)
print("The topics described by their top-weighted terms:")

display(topics)


# In[17]:


# extract vocabulary from CountVectorizer
vocab = vectorizer.fit(df_preprocess).vocabulary
topics = ldaModel.describeTopics()   
topics_rdd = topics.rdd
topics_words = topics_rdd       .map(lambda row: row['termIndices'])       .map(lambda idx_list: [vocab[idx] for idx in idx_list])       .collect()
for idx, topic in enumerate(topics_words):
    print("topic: {}".format(idx))
    print("*"*25)
    for word in topic:
       print(word)
    print("*"*25)


# In[18]:


print(vocab)

