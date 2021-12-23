#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import necessary packages
import pandas as pd
import numpy as np

import os 
import matplotlib
import matplotlib.pyplot as plt


# In[ ]:


#import train and test data
train = pd.read_csv('/Users/williamchan/Desktop/sentiment_train.csv')
test = pd.read_csv('/Users/williamchan/Desktop/sentiment_test.csv')


# In[ ]:


#combined dfs for pre-processing
train['Type'] = 'train'
test['Type'] = 'test'


# In[ ]:


#combined dfs for pre-processing
frames = [train, test]
df = pd.concat(frames)


# In[ ]:


#generate text features
import textstat

def preprocess(x):
    return textstat.flesch_reading_ease(x)
    
df['len'] = df['Sentence'].apply(lambda x: len(x))
df['syllable_count'] = df['Sentence'].apply(lambda x: textstat.syllable_count(x))
df['lexicon_count'] = df['Sentence'].apply(lambda x: textstat.lexicon_count(x))
df['sentence_count'] = df['Sentence'].apply(lambda x: textstat.sentence_count(x))
df['flesch_reading_ease'] = df['Sentence'].apply(lambda x: textstat.flesch_reading_ease(x))
df['flesch_kincaid_grade'] = df['Sentence'].apply(lambda x: textstat.flesch_kincaid_grade(x))
df['gunning_fog'] = df['Sentence'].apply(lambda x: textstat.gunning_fog(x))
df['smog_index'] = df['Sentence'].apply(lambda x: textstat.smog_index(x))
df['coleman_liau_index'] = df['Sentence'].apply(lambda x: textstat.coleman_liau_index(x))
df['automated_readability_index'] = df['Sentence'].apply(lambda x: textstat.automated_readability_index(x))
df['dale_chall_readability_score'] = df['Sentence'].apply(lambda x: textstat.dale_chall_readability_score(x))
df['difficult_words'] = df['Sentence'].apply(lambda x: textstat.difficult_words(x))
df['linsear_write_formula'] = df['Sentence'].apply(lambda x: textstat.linsear_write_formula(x))
#%time df['text_standard'] = df['Sentence'].apply(lambda x: textstat.text_standard(x))


# In[ ]:


#check df
df.shape


# In[ ]:


#generate text features
#here we manually tuned min_df, ngram_range, and max_features
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=15, ngram_range=[1,2], max_features=1000, stop_words='english')
dtm_tfidf = tfidf_vectorizer.fit_transform(df['Sentence'])

bow_df_tfidf = pd.DataFrame(dtm_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names(), index=df.index)

df_bow_tfidf = pd.concat([df, bow_df_tfidf], axis=1)
df_bow_tfidf.drop(columns=['Sentence'], inplace=True)
df_bow_tfidf.shape
df_bow_tfidf.head()


# In[ ]:


#split data after pre-processing for modeling
test_final = df_bow_tfidf[df_bow_tfidf['Type'] == 'test']
train_final = df_bow_tfidf[df_bow_tfidf['Type'] == 'train']


# In[ ]:


#check data
test_final.shape


# In[ ]:


train_final.shape


# In[ ]:


#drop type column
train_final = train_final.drop(columns=['Type'])
test_final = test_final.drop(columns=['Type'])


# In[ ]:


y_train = train_final['Polarity']
y_test = test_final['Polarity']
X_train = train_final.drop(columns=['Polarity'])
X_test = test_final.drop(columns=['Polarity'])


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


#import sklearn package and run Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#Scale the data to minimize extremes
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Run decision tree classifier
clf = DecisionTreeClassifier(random_state=42, min_samples_split=10, min_samples_leaf=10, max_depth=6)
clf.fit(X_train, y_train)


# In[ ]:


#Predict on test data
clf_predition = clf.predict(X_test)


# In[ ]:


#get feature importances
clf.feature_importances_


# In[ ]:


#compute feature importances
imp = clf.tree_.compute_feature_importances(normalize=False)
ind = sorted(range(len(imp)), key=lambda i: imp[i])[-15:]

imp[ind]
feature_names[ind]


# In[ ]:


#print confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, clf_predition)


# In[ ]:


#print decision tree classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, clf_predition))


# In[ ]:


#plot tree
from sklearn.tree import plot_tree

plt.figure(figsize=[20,10]);
plot_tree(clf, filled=True, feature_names = feature_names, label='root', fontsize=14)
plt.show();


# In[ ]:


#compare other models
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB

print("NB")

gnb = GaussianNB()
get_ipython().run_line_magic('time', 'gnb = gnb.fit(X_train, y_train)')

y_pred_nb = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

gnb.theta_


print("\n\nKNN")
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=15)
knn_clf.fit(X_train, y_train)

y_pred_knn = knn_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


print("\n\nRF")
clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=55)
get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')

imp = pd.DataFrame(clf.feature_importances_, index = feature_names, columns=['importance']).sort_values('importance', ascending=False).iloc[0:15,:]
print(imp)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\n\nAdaboost")
ada_clf = AdaBoostClassifier(n_estimators=200)
get_ipython().run_line_magic('time', 'ada_clf.fit(X_train, y_train)')
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\n\nGBC")
gbc_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=45)
get_ipython().run_line_magic('time', 'gbc_clf.fit(X_train, y_train)')
y_pred = gbc_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\n\nHGBC")
hgbc_clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
hgbc_clf.score(X_test, y_test)
y_pred = hgbc_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


#Select Gradient Boosting Classifer as the model to tune
gbc_clf = GradientBoostingClassifier(random_state=45)


# In[ ]:


#Setup Parameter Gridsearch features
param_grid = { 
'learning_rate': [0.01,0.02,0.03],
'subsample'    : [0.9, 0.5, 0.2],
'n_estimators' : [100,500,1000],
'max_depth'    : [4,6,8]
}


# In[ ]:


#Gridsearch to figure out optimal parameters
from sklearn.model_selection import GridSearchCV

gbc_clf = GridSearchCV(estimator=gbc_clf, param_grid=param_grid, cv= 3, verbose=10)
gbc_clf.fit(X_train, y_train)


# In[ ]:


#See best parameters
gbc_clf.best_params_


# In[ ]:


#Select GBC as final model and enter best parameters from GridSearch to tune

print("\n\nGBC")
gbc_clf = GradientBoostingClassifier(random_state=45, learning_rate = 0.05, n_estimators = 500, max_depth = 12, subsample = 0.9)
get_ipython().run_line_magic('time', 'gbc_clf.fit(X_train, y_train)')
y_pred = gbc_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

imp = pd.DataFrame(gbc_clf.feature_importances_, index = feature_names, columns=['importance']).sort_values('importance', ascending=False).iloc[0:20,:]
print(imp)


# In[ ]:


#Export predictions
np.savetxt("final_predictions.csv", y_pred, delimiter=",")

