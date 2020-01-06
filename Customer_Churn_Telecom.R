#PREDICTING CUSTOMER CHURN IN TELECOM INDUSTRY
#04/01/2020

#Cell Phone Data
library(caret) 
library(ggplot2) 
#library(Information) 
#library(caTools) 
#library(stringr) 
#library(car) 
#library(ROCR) 
#ibrary(MASS) 
#library(gmodels) 
#library(dummies) 
#library(Hmisc)
library(readxl)
library(lmtest)
library(pscl)
library(tidyverse)


setwd("/Users/apple/MEGA/Personal/My_Projects/DS_Projects/customer_churn")
## Import Data
CellData <- read.csv("Dataset_Cellphone.csv")
#CellData <- read.csv("telecom_churn.csv")

str(CellData)
summary(CellData)
head(CellData)
tail(CellData)

#This dataset contains one dependent variable and 10 independent variable
#Our aim is to identify the predictor variables which are significant for Customer Churn.

# Null Hypothesis : No predictor is able to predict the churn
# Alternate Hypothesis: At least one predictor is able to predict the churn

#Step 1: Check for Missing Values
sum(is.na(CellData))

#Step 2: Convert as factor variables & perform Outlier Treatment
#converting churn as factor variable
table(CellData$Churn)
CellData$Churn <- as.factor(CellData$Churn)
levels(CellData$Churn) <- c("NotCancelled","Cancelled")
str(CellData$Churn)

#converting ContractRenewal as factor variable
table(CellData$ContractRenewal)
CellData$ContractRenewal <- as.factor(CellData$ContractRenewal)
levels(CellData$ContractRenewal) <- c("Not Renewed","Recently Renewed")
str(CellData$ContractRenewal)


#converting Data Plan as factor variable
table(CellData$DataPlan)
CellData$DataPlan <- as.factor(CellData$DataPlan)
levels(CellData$DataPlan) <- c("No Data Plan","Has Data Plan")
str(CellData$DataPlan)

#Outlier Treatment | Variable: AccountWeeks
quantile(CellData$AccountWeeks,c(0.05,0.1,0.2,0.3,0.4,0.50,0.6,0.7,0.8,0.9,0.95,0.99,1))
#limiting to 195 (99% values)
CellData$AccountWeeks[which(CellData$AccountWeeks > 195)] <- 195
summary(CellData$AccountWeeks)

# Outlier Treatment | Variable: DataUsage

quantile(CellData$DataUsage,c(0.05,0.1,0.2,0.3,0.4,0.50,0.6,0.7,0.8,0.9,0.95,0.99,1))
#limiting to 4.10 (99% values)
CellData$DataUsage[which(CellData$DataUsage > 4.10)] <- 4.10
summary(CellData$DataUsage)

# Outlier Treatment | Variable: CustServCalls
quantile(CellData$CustServCalls,c(0.05,0.1,0.2,0.3,0.4,0.50,0.6,0.7,0.8,0.9,0.95,0.99,1))
#limiting to 6 (99% values)
CellData$CustServCalls[which(CellData$CustServCalls > 6)] <- 6
summary(CellData$CustServCalls)


#DayMins DayCalls MonthlyCharge RoamMins OverageFee
#- all these avariables seem to be in acceptable ranges without any outliers 
# not limiting any of them

# Step 3: Create Dummy variable for all categorical variables
summary(CellData)
CellData <- dummy.data.frame(CellData, names = c("Churn","ContractRenewal","DataPlan"), sep = "_")
colnames(CellData)
#We will remove one dummy variable from each categorical variables so that we have (n-1)
#dummy variables
CellData <- CellData[,-c(2,5,7)]
colnames(CellData)

#Step 4: Divide Data into 70 : 30, 
# On 70% data we will build the model, on 30% data we will train the model
set.seed(150)
split_indices <- sample.split(CellData, SplitRatio = 0.70)
train <- CellData[split_indices,]
test <- CellData[!split_indices,]

#Step 5: Build logistic regression model with stepwise variable selection
#MODELLING LOG Regression
logistic_1 <- glm(Churn_NotCancelled ~ ., family = "binomial", data = train)
summary(logistic_1)
#the model shows 3 variables are significant in predicting the Churn, ContractRenewal_Not_Renewed (Negative impact)
#CustServCalls & RoamMins
#Note: AIC score: 1383.2
# The model having least AIC Score would be the most preferred and optimized one.
#AIC: The Akaike information criterion (AIC) is a measure of the relative quality of statistical models for a given set of data. 
#Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. 
#Given a set of candidate models for the data, the preferred model is the one with the minimum AIC value.
#AIC rewards goodness of fit (as assessed by the likelihood function), 
#but it also includes a penalty that is an increasing function of the number of estimated parameters

#Checking Model Significance: using Log Likelyhood Ratio
lrtest(logistic_1)

#Interpretation:
#H0: All betas are zero
#H1: At least 1 beta is nonzero
#From the log likelihood, we can see that, intercept only model -859.04 variance was unknown to us. 
#When we take the full model, -680.61 variance was unknown to us. 
#So we can say that, 1 – (-680.61 /- 859.04)= 20.77% of the uncertainty inherent in the 
#intercept only model is calibrated by the full model.

#Test Model Robustness
pR2(logistic_1)

#The McFadden’s pseudo-R Squared test suggests that atleast 20.77% variance of the data 
#is captured by our Model, which suggests it’s a robust model.


#Lets run stepwise selection of variables, based on the significance of variables
logistic_2 <- stepAIC(logistic_1, direction = "both",k = 3)
summary(logistic_2)

#Remove the variable which has only one and 2 stars, retain only 3 star variables.
#Look at the |Z| value of the variable. Higher the |Z| value of the variable, that one is the most significant variable

logistic_3 <- glm(Churn_NotCancelled ~ 
                    `ContractRenewal_Not Renewed`
                  + DataUsage
                  + CustServCalls
                  + MonthlyCharge
                  + RoamMins
                  , family = "binomial", data = train)
summary(logistic_3)

#Step 6: Check for Multicollinearity, using Value inflation factor (VIF)

vif(logistic_3)
logistic_final <- logistic_3

#Model Performance Measures
#- Confusion Matrix
#- Predicting probabilities of responding for the training data

train$predicted_prob = predict(logistic_final,  type = "response")
train$predicted_response <- factor(ifelse(train$predicted_prob >= 0.2, "yes", "no"))
train$newchurn<-as.factor(ifelse(train$Churn=="1","yes","no"))
conf <- confusionMatrix(train$predicted_response, train$newchurn, positive = "yes")
conf

#Plotting the ROC Curve

predicted_response_lr <- ifelse(train$predicted_response == "yes",1,0)
actual_response_lr <- ifelse(train$newchurn == "yes",1,0)
model_score_test_lr <- prediction(predicted_response_lr,actual_response_lr)
model_perf_test_lr <- performance(model_score_test_lr, "tpr", "fpr")

plot(model_perf_test_lr,col = "green", lab = c(10,10,10))

#Checking the KS Statistics on Train dataset

kstable <- attr(model_perf_test_lr, "y.values")[[1]] - (attr(model_perf_test_lr, "x.values")[[1]])
ks = max(kstable)
ks

# Predicting probabilities of responding for the test data
test$predicted_prob <- predict(logistic_final, newdata = test, type = "response")
# Creating confusion matrix for test dataset.
test$predicted_response <- factor(ifelse(test$predicted_prob >= 0.2, "yes", "no"))
test$newchurn <- as.factor(ifelse(test$Churn == "1","yes","no"))
conf <- confusionMatrix(test$predicted_response, test$newchurn, positive = "yes")
conf
#INterpretation:
#1024 out of (1024+183) customeres identified sucessfully which have been churned out
#ie 84% of positive prediction value
#3 out of (3+2) customers identified sucessfully whic have not been churned out
# this translates to 60% negative prediction value
# At 84.74% model provides good accuracy measure
#

#Plotting the ROC Curve on Test Data

predicted_response_lr_test <- ifelse(test$predicted_response == "yes",1,0)
actual_response_lr_test <- ifelse(test$newchurn == "yes",1,0)
model_score_test_lr <- prediction(predicted_response_lr_test,actual_response_lr_test)
model_perf_test_lr <- performance(model_score_test_lr, "tpr", "fpr")

plot(model_perf_test_lr,col = "green", lab = c(10,10,10))
library(deducer)
rocplot(logistic_final)
#Checking the KS Statistics on test dataset
kstable <- attr(model_perf_test_lr, "y.values")[[1]] - (attr(model_perf_test_lr, "x.values")[[1]])
ks = max(kstable)
ks
