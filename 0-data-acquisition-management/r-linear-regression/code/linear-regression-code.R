#ASSIGNMENT 2 - QUESTION 1

#PROBLEM D

#IMPORT LIBRARIES
library(readxl)
library(tidyverse)
library(mice)

#IMPORT DATA
missing <- read_excel("OneDrive - Queen's University/Global Master of Management Analytics/GMMA 860 - Acquisition and Management of Data/Assignment #2/GMMA860_Assignment2_Data.xlsx", sheet = "Missing")

#VIEW SUMMARIES
head(missing)
summary(missing)

#CREATE THE REGRESSION MODEL
reg <- lm(Y ~ X1 + X2 + X3 + X4 + X5, missing)

#VIEW THE SUMMARY
summary(reg)

#IMPUTE DATA
imputed <- mice(missing, m=5, maxit=30, meth='pmm', seed=1)

#CREATE REGRESSION WITH IMPUTED DATA
reg1 <- with(imputed, lm(Y ~ X1 + X2 + X3 + X4 + X5))
summary(reg1)
summary(pool(reg1))

#ASSIGNMENT 2 - QUESTION 2

#PROBLEM A

#IMPORT DATA
missing <- read_excel("OneDrive - Queen's University/Global Master of Management Analytics/GMMA 860 - Acquisition and Management of Data/Assignment #2/GMMA860_Assignment2_Data.xlsx", sheet = "Wine")

#VIEW SUMMARY
head(wine)
summary(wine)

#DUMMY THE VARIABLES
wine$Canada <- ifelse(wine$Country == "Canada", 1, 0) 
wine$US <- ifelse(wine$Country == "US", 1, 0) 
wine$France <- ifelse(wine$Country == "France", 1, 0) 
wine$Italy <- ifelse(wine$Country == "Italy", 1, 0)

#VIEW AND CONFIRM DATA
view(wine)

#RUN FIRST REGRESSION MODEL
reg_wine <- lm(Rating ~ Price + Alcohol + Residual_Sugar + Sulphates + pH + Canada + US + Italy + France + US, wine)

#ANALYZE PLOTS
summary(reg_wine)
plot(reg_wine)

#Second Reg Model - FINAL MODEL
reg_wine1 <- lm(Rating ~ Price + Alcohol + Sulphates + Canada + US, wine)
summary(reg_wine1)
plot(reg_wine1)

#PROBLEM C

#RUN MODEL WITH 4 VARIABLES
reg_wine2 <- lm(Rating ~ Price + Alcohol + Sulphates + France, wine)
summary(reg_wine2)
plot(reg_wine2)

