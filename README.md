# Practical-Machine-Learning-Write-Up

This ripo contains the write up part of the Practical Machine Learning project work. The files included in this repo are:

1) [R mark down document](./Practical Machine Learning - Assi 1.Rmd), 

2) [HTML document] (./Practical_Machine_Learning_-_Assi_1), and 

3) [Submission] (./Submssion) files that contain the answer for the test data. 


---
title: "Classification of Personal Activity using Machine Learning Algorithms"
author: "Filmon G. Habtemichael"
date: "Tuesday, August 18, 2015"
output: html_document
---

This document was prepared as a response to a homework assignment the course Practical Machine Learning offered from Johns Hopkins University in the Data Science Specialization. 

## Introduction

The objective of this document is to apply machine learning algorithms in an attempt to classify personal activity information collected from nomadic devices (activities include several kinds of weight lifting excersices). In this project, several machine learning algorithms are employes, namely: **Decision Tree** and **Random Forest**. A link to Github repo containg R markdown and compiled HTML files describing the analysis performed in this project are also provied. 

"Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)." 


## Data Sources

The data sets used in this analysis were obtained from a project that focused on Human Activity Recognition (HAR). More information about the data sets can be obtained from the web page http://groupware.les.inf.puc-rio.br/har. The author would like the HAR group for graciously allowing us to use the data set.

The trainig and testing data sets were made available in the following URLs: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


## Objectives of this project 

The goal of this project is to predict the manner in which individuals performed their exercise. This is the "classe" variable in the training set. To predict the specific manner of excercise the individuals performed, several variables were used as covariants. 

This report provides information on the following important points on machine learning algorithms:

-How the predictive model is developed?
-How the results were cross validated?
-What the expected out of sample error is?
-Apply the prediction model to predic 20 different test cases. 


## Reproduceablity

For the results if this project to be reproducible, a pseudo seed number of 12 is used thoughout the analysis. 

```{r}
set.seed(22)
````

Loding the required libraries in R

```{r, warning=FALSE}
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(e1071)
```

## Loading the data

The training and testing data set can be found on the following URL:

```{r}
Url_Testing <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
Url_Training <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
```

Loading the data to memory:
From an initial exploratoty analysis, the several entries for missing data were noted. This include: "NA", "#DIV/0!" and "". Therefore all such entries in the training and testingg data sets are coded as "NA" in this project.

```{r}
training <- read.csv(url(Url_Training), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(Url_Testing), na.strings=c("NA","#DIV/0!",""))
```

Note that the above code lines will download the data into the current local directory (hard drive).


## Dividing the data into TESTING and TRAINING data sets

In data analysis, it is very important that a model be trained in a training data set and tested its performance using an indipendent data sets which are called testing data sets. Given a data sets, it is customary to assign 60% of the data as a trainig and the rest 40% as a testing data sets. Since the data sets in this project are indipend observations, the training and testing data sets can be created by random sampling (had the data being time series, trainig and testing data sets would have been constructed by considering chunkes of the data).

```{r}
Sample_Training <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
Training_Data <- training[Sample_Training, ]; Testing_Data <- training[-Sample_Training, ]
dim(Training_Data); dim(Testing_Data)
```


## Cleaning and preparing the data

The following transformations were used to clean the data:

**Transformation 1:** Cleaning NearZeroVariance Variables Run this code to view possible NZV Variables:

```{r}
myDataNZV <- nearZeroVar(Training_Data, saveMetrics=TRUE)
```

Run this code to create another subset without NZV variables:

```{r}
myNZVvars <- names(Training_Data) %in% c("new_window", "kurtosis_roll_belt", "kurtosis_picth_belt", "kurtosis_yaw_belt", "skewness_roll_belt", "skewness_roll_belt.1", "skewness_yaw_belt", "max_yaw_belt", "min_yaw_belt", "amplitude_yaw_belt", "avg_roll_arm", "stddev_roll_arm", "var_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm", "stddev_yaw_arm", "var_yaw_arm", "kurtosis_roll_arm", "kurtosis_picth_arm", "kurtosis_yaw_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm","max_roll_arm", "min_roll_arm", "min_pitch_arm", "amplitude_roll_arm", "amplitude_pitch_arm", "kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "kurtosis_yaw_dumbbell", "skewness_roll_dumbbell", "skewness_pitch_dumbbell", "skewness_yaw_dumbbell", "max_yaw_dumbbell", "min_yaw_dumbbell", "amplitude_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "kurtosis_yaw_forearm", "skewness_roll_forearm", "skewness_pitch_forearm", "skewness_yaw_forearm", "max_roll_forearm", "max_yaw_forearm", "min_roll_forearm", "min_yaw_forearm", "amplitude_roll_forearm", "amplitude_yaw_forearm", "avg_roll_forearm", "stddev_roll_forearm", "var_roll_forearm", "avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm", "stddev_yaw_forearm", "var_yaw_forearm")
Training_Data <- Training_Data[!myNZVvars]
#To check the new No. of observations
dim(Training_Data)
```

**Transformation 2:** Removing unnecessary variables. The first column of Dataset, i.e., ID, is not a predictive variable and thus should be removed from the data so that it doesn't interfere with the predictive model to be developed.

```{r}
Training_Data <- Training_Data[c(-1)]
```

**Transformation 3:** Cleaning Variables with too many NAs. For Variables that have more than a 60% threshold of NA's I'm going to leave them out:

```{r}
trainingV3 <- Training_Data ##creating another subset to iterate in loop
for(i in 1:length(Training_Data)) { ##for every column in the training dataset
    if( sum( is.na( Training_Data[, i] ) ) /nrow(Training_Data) >= .6 ) { ##if no. NAs > 60% of total observations
        for(j in 1:length(trainingV3)) {
            if( length( grep(names(Training_Data[i]), names(trainingV3)[j]) ) ==1)  { ##if the columns are the same:
                trainingV3 <- trainingV3[ , -j] # Removes the unneed column
}}}}

Training_Data <- trainingV3
rm(trainingV3)
```

The testing datasets should also be treated with the same transformation applied to the training data set. The following code will perform that:

```{r}
clean1 <- colnames(Training_Data)
clean2 <- colnames(Training_Data[, -58]) 
Testing_Data <- Testing_Data[clean1]
testing <- testing[clean2]
```

In order to ensure proper functioning of Decision Trees and especially RandomForest Algorithm with the Test data set (data set provided), we need to coerce the data into the same type.

```{r}
for (i in 1:length(testing) ) {
        for(j in 1:length(Training_Data)) {
        if( length( grep(names(Training_Data[i]), names(testing)[j]) ) ==1)  {
            class(testing[j]) <- class(Training_Data[i])
}}}      
##And to make sure Coertion really worked:
testing <- rbind(Training_Data[2, -58] , testing) ##If column names are the same, there will be no error message. 
testing <- testing[-1,] # remove the binded row which was actually used to check the coertion
```

## Predictions using Machine Learning algorithms

Now that our data are clean and tidy, they are for applying machine learning algorithms. Two machine learning algorithms are applied in this project, namely: **Decision Tree** and **Random Forest**


### PREDICTION USING DECISION TREE

The following code fits a decision tree model to our training data set. The resulting decision tree is also shown.

```{r}
Decision_T_Fit <- rpart(classe ~ ., data=Training_Data, method="class")
fancyRpartPlot(Decision_T_Fit)
```

Now we have our decision tree model developed, it can be used to predict the activity of individuals. Let's test the accuracy of our model using our training data set. The confusin matrix is also provided below.

```{r}
predictionsA1 <- predict(Decision_T_Fit, Testing_Data, type = "class")

confusionMatrix(predictionsA1, Testing_Data$classe)
```


### PREDICTION USING RANDOM FOREST

The following code fits random forest to the training data sets and also checks the accuracy of the modeling.

```{r}
Random_F_Fit <- randomForest(classe ~. , data=Training_Data)

predictionsB1 <- predict(Random_F_Fit, Testing_Data, type = "class")

confusionMatrix(predictionsB1, Testing_Data$classe)
```


## Comparing the results of the Decision Tree and the Random Forest

Based on the resulting confusion matrices, the results from the Decision Tree and Random Forest algorithms are very good - **the out of sample error for the models was 86% and 99.9% respectively**. Although both models are comparable, the results from the Random Forest algorithm are better than the result from the Decision Tree.   

## Conclusions

In this project, machine learning algorithms were applied to predict the human activities in lated to weight lifting. Two models were developed, namely: Decision Tree and Random Forest. The results of the experiment are very incouraging as the activities can be accurately estimated with out of sample error of 99.9%.  

## Generating Files to submit as answers for the Assignment:

The predictions for the assignment are provided using Random Forest algorithm that was developed in the previsous sections. The following code provides answer to the asignment and generates the answers in separate files for automatic grading purposes.

```{r}
predictionsB2 <- predict(Random_F_Fit, testing, type = "class")
```

Generating separate files as an answerr to the assignment. 

```{r}
setwd("D:/Practical Machine Learning - Assignment")
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictionsB2)

```
