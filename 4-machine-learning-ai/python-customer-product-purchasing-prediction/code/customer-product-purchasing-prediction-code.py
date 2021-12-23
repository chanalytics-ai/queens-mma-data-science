# [William Chak Lim Chan]
# [20198113]
# [GMMA]
# [Inaugural]
# [GMMA 869]
# [July 5, 2020]

# Answer to Question [7], Part [2a]

# # a. Preprocess the data however you see fit
# # Import Packages

#Install PyCaret
pip install pycaret

#Install Missing LightGBM Package
conda install -c conda-forge/label/cf202003 lightgbm

#Import Pandas and Pandas Profiling for Data Handling
import pandas as pd
import pandas_profiling

# # Read .csv Data

#Import Data
df = pd.read_csv('/Users/williamchan/Desktop/oj.csv')

# # Data Profiling

# Understand the data shape
list(df)
df.shape
df.info()
df.describe().transpose()
df.head(n=10)
df.tail()

# Understand the data - pull stats
pandas_profiling.ProfileReport(df)

#Drop columns due to high correlation between PctDiscMM and PctDiscCH
df1 = df.drop(columns = ['PctDiscMM', 'PctDiscCH'])

#Verify column was dropped
df1.info()

# # b. Split the data into training and testing sets
# # c. Build three different models, using three different machine learning algorithms. Tune. Print. Test and Print.
# # Setting up Environment

# Answer to Question [7], Part [2b & 2c]

# Importing a Module
from pycaret.classification import *

#Setup PyCaret, split test and train using 75% for training to ensure a solid size for test.
clf = setup(data = df1, target = 'Purchase', train_size = 0.75, session_id = 2986)

# # Run all models as part of PyCaret package and compare Models using performance metrics

# comparing all models
compare_models(sort = 'Precision')

# # Create 3 Models based on highest Precision

#Create Logistic Regression Model
lr = create_model('lr')

#Create Linear Discriminant Analysis Model
lda = create_model('lda')

#Create Extreme Gradient Boosting Model
xgboost = create_model('xgboost')

# # Tune 3 Models to optimize for Precision

tuned_lr = tune_model('lr', optimize = 'Precision')

tuned_lda = tune_model('lda', optimize = 'Precision')

tuned_xgboost = tune_model('xgboost', optimize = 'Precision')

# # Print Tuned Hyper Parameters

print(lr)

print(tuned_lr)

print(lda)

print(tuned_lda)

print(xgboost)

print(tuned_xgboost)

# # Review Model Performance

evaluate_model(lr)

evaluate_model(lda)

evaluate_model(xgboost)

# # Predict Using Test Data and Print Results

lr_predictions_test = predict_model(tuned_lr)

finalize_model(tuned_lr)

lda_predictions_test = predict_model(tuned_lda)

finalize_model(tuned_lda)

xgboost_predictions_test = predict_model(tuned_xgboost)

finalize_model(tuned_xgboost)



