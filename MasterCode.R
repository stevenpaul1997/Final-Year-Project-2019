---
  title: "AMExpert 2018 Machine Learning Hackathon"
Author: "JOSEPH OLADOKUN"
output: html_notebook
---
  library(ggplot2)
library(dplyr)
library(rpart)
library(ROSE)
library(rattle)
library(pROC)
library(caret)
library(mlbench)
library(caretEnsemble)
library(randomForest)


##Import Data Set

train <- read.csv("train.csv")
test <- read.csv("test.csv")

##Insert a new column "is_click" in test

test$is_click <- NA

## Briefly look at the data

dim(train)
str(train)
str(test)

## Let's Drop the Column "DateTime" because it has too many factor levels and product category 2 bacause it has over 90% missing values

train$DateTime <- NULL
train$product_category_2 <- NULL
test$DateTime <- NULL
test$product_category_2 <- NULL

## Factorising "is_click" and replacing 1 and 0 to "Yes", "No" respectively. 

train$is_click[train$is_click==1]<-"Yes"
train$is_click[train$is_click==0]<-"No"
train$is_click<-as.factor(train$is_click)
test$is_click<- as.factor(test$is_click)

## Visualising the diffrence between "Click and No Click"

table(train$is_click)
v1<- ggplot(train, aes(is_click))+geom_bar()+ggtitle( "Click vs. No Click")
v1

##....................................................................................................................................
## First let us see the performance of randomforest model on the data (imbalanced data). This will allow us to observe the improvement with the sampling later on.


tree_model <- randomForest(is_click ~ .,  data=train, na.action = na.exclude)
pred.tree_model <- predict(tree_model, newdata = test)

## Plotting the ROC Curve
roc.curve(test$is_click, pred.tree_model[,2], plotit = F)

##Writing out the result
data.frame(pred.tree_model)
write.csv(pred.tree_model, file = "output.csv")

##It can be observed that the dataset is highly imbalanced with only 31331 Clicks out of total 463291 observations. To work with such data we have to explore ##various sampling methods to achieve meaningfull level of observations in both categories. Here we will apply:
##    Undersampling - working with majority class (No) to decrease it's number to the level of minority class (Yes). The shortcoming here is the data loss
##    Oversampling - working with minority class (Yes) to increase its number to the level of majority class (No). The shortcoming here is that it leads to overfitting
##   Synthetic Data Generation - this is to balance our dataset i.e Working with minority and majority classes to make 50/50 balance

## Let's Undersample the Data
train_undersample <- ovun.sample(is_click ~ ., data = train, method = "under", N = 984, seed = 1)$data
table(train_undersample$is_click)

## Oversampling
train_oversample <- ovun.sample(is_click ~ ., data = train, method = "over", N = 568630)$data
table(train_oversample$is_click)

##Both
train_both <- ovun.sample(is_click ~ ., data = train, method = "both", p=0.5, N = 10000, seed = 1)$data
table(train_both$is_click)

##Synthetic Data Generation
train_rose <- ROSE(is_click ~ ., data = train, seed = 1)$data
table(train_rose$is_click)

##CART Model with resampled data
fit.under <- rpart(is_click ~ ., data = train_undersample)
fit.over <- rpart(is_click ~ ., data = train_oversample)
fit.both <- rpart(is_click ~ ., data = train_both)
fit.rose <- rpart(is_click ~ ., data = train_rose)

##CART Predictions on resampled data
pred.under <- rpart(is_click ~ ., data = train_undersample)
pred.over <- rpart(is_click ~ ., data = train_oversample)
pred.both <- rpart(is_click ~ ., data = train_both)
pred.rose <- rpart(is_click ~ ., data = train_rose)

##CART AUC on resampling methods vs. original data AUC
roc_und<-plot(roc(test$is_click,pred.under[,2]), print.auc=TRUE,col="blue")
roc_over<-plot(roc(test$is_click, pred.over[,2]), print.auc=TRUE, col="green", print.auc.y=.4, add=TRUE)
roc_both<-plot(roc(test$is_click, pred.both[,2]), print.auc=TRUE, col="red", print.auc.y=.6, add=TRUE)
roc_rose<-plot(roc(test$is_click, pred.rose[,2]), print.auc=TRUE, col="orange", print.auc.y=.5, add=TRUE)
roc_skewed<-plot(roc(test$is_click, pred.tree_model[,2]), print.auc=TRUE, col="black", print.auc.y=.7, add=TRUE)

##Let's Apply ML Models
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('rpart','glm','knn','svmRadial')
set.seed(2)
models <- caretList(Class~., data=fraud_balanced_both1, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

##In terms of accuracy the best result belongs to SVM

##Model Correlations
modelCor(results) 
splom(results)

##Let's Stack all 4 models.
##Model Stacking with GLM
set.seed(3)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=control)
print(stack.glm)
