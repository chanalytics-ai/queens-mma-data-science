#load data
df <- read.csv(file.choose())

# basic descriptive statistics 
summary(df)
gpairs(df)
corrplot.mixed(cor(df[ , c(1, 2:8)]), upper="ellipse")

#split the data into training and test sample

train.df<-df[1:750,]
test.df<-df[751:1000,]

#model 1 - train
m1.train <- lm(sales ~ price, data=train.df)
summary(m1.train)
par(mfrow=c(2,2))
plot(m1.train)

#model 1 - test
m1.test <- predict(m1.train,test.df)

# Compute R-squared in the test sample 
# R-squared = Explained variation / Total variation
SSE = sum((test.df$sales - m1.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

#model 2 - train
m2.train <- lm(sales ~ price + store, data=train.df)
summary(m2.train)
plot(m2.train)

#model 2 - test
m2.test <- predict(m2.train,test.df)
SSE = sum((test.df$sales - m2.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

anova(m1.train, m2.train)

#model 3 - train
m3.train <- lm(sales ~ price + store + billboard, data=train.df)
summary(m3.train)
plot(m3.train)

#model 3 - test
m3.test <- predict(m3.train,test.df)
SSE = sum((test.df$sales - m3.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

anova(m2.train, m3.train)

#model 4 - train
m4.train <- lm(sales ~ price + store + billboard + printout, data=train.df)
summary(m4.train)
plot(m4.train)

#model 4 - test
m4.test <- predict(m4.train,test.df)
SSE = sum((test.df$sales - m4.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

anova(m3.train, m4.train)

#model 5 - train
m5.train <- lm(sales ~ price + store + billboard + printout + sat, data=train.df)
summary(m5.train)
par(mfrow=c(2,2))
plot(m5.train)

#model 5 - test
m5.test <- predict(m5.train,test.df)
SSE = sum((test.df$sales - m5.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

anova(m4.train, m5.train)

#model 6 - train
m6.train <- lm(sales ~ price + store + billboard + printout + sat + comp, data=train.df)
summary(m6.train)
par(mfrow=c(2,2))
plot(m6.train)

#model 6 - test
m6.test <- predict(m6.train,test.df)
SSE = sum((test.df$sales - m6.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

anova(m5.train, m6.train)

#model 7 - train
m7.train <- lm(sales ~ price + store + billboard + printout + sat + comp + store:billboard + store:printout + billboard:printout, data=train.df)
summary(m7.train)
par(mfrow=c(2,2))
plot(m7.train)

#model 7 - test
m7.test <- predict(m7.train,test.df)
SSE = sum((test.df$sales - m7.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

anova(m6.train, m7.train)

#model 8 - train - final model
m8.train <- lm(sales ~ price + store + billboard + printout +  sat + comp + store:billboard, data=train.df)
summary(m8.train)
par(mfrow=c(2,2))
plot(m8.train)

#model 8 - test - final model
m8.test <- predict(m8.train,test.df)
SSE = sum((test.df$sales - m8.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

anova(m7.train, m8.train)

#model 9 - train - final model
m9.train <- lm(sales ~ price + billboard + store +  sat + comp + store:billboard, data=train.df)
summary(m9.train)
par(mfrow=c(2,2))
plot(m9.train)

#model 9 - test - final model
m9.test <- predict(m9.train,test.df)
SSE = sum((test.df$sales - m9.test)^2) # Explained variation
SST = sum((test.df$sales - mean(test.df$sales))^2) # Total Variation
Rsq = 1 - SSE/SST
Rsq

anova(m8.train, m9.train)
anova(m7.train, m9.train)

coefplot(m9.train, intercept=FALSE, outerCI=2, lwdOuter=1.5,
         ylab="Internal Spend Effects and Competitor Pricing Features", 
         xlab="Association with Overall Sales")

par(mfrow=c(1,1))
plot(train.df$sales, fitted(m9.train), col='red', xlim=c(0,30000), ylim=c(0,30000), xlab="Actual Sales", ylab="Fitted Sales")
points(train.df$sales, fitted(m1.train), col='blue')
legend("bottomright", legend=c("model 9", "model 1"), 
       col=c("red", "blue"), pch=1)
