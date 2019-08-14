################################################################################################
#### This code is for the Higgs Boson Kaggle Competition
## Created: August 13, 2019
## Edited:
################################################################################################

rm(list = ls())

# set working directory
setwd("/Users/m/Desktop/Kaggle/Higgs Boson")

# video url: https://www.youtube.com/watch?v=ufHo8vbk6g4
# Higgs Boson Data from Kaggle: https://www.kaggle.com/c/higgs-boson/data

library(xgboost)
library(methods)
library(DiagrammeR)




test <- read.csv("test.csv", header=T)
train <- read.csv("training.csv", header=T)
sub <- read.csv("random_submission.csv", header=T)

test <- test[1:250000,]


################################################################################################
# Higgs Boson Competition
################################################################################################
# set our test size
testsize = 550000

# change train[33] to a binary variable and define necessary parameters for the XGBoost
train[33] = train[33] == "s"
label = as.numeric(train[[33]])
data = as.matrix(train[2:31])
weight = as.numeric(train[[32]]) * testsize / length(label)
sumwpos <- sum(weight*(label==1.0))
sumwneg <- sum(weight*(label==0.0))

# construct an xgb.DMatrix conataining the info of weight and missing
xgmat = xgb.DMatrix(data, label=label, weight=weight, missing=-999.0)

# define the basic parameters
param = list('objective' = 'binary:logitraw',
             'scale_pos_weight' = sumwneg / sumwpos,
             "bst:eta" = 0.1,
             'bst:max_depth' = 6,
             'eval_metric' = 'auc',
             'eval_metric' = 'ams@0.15',
             'silent' = 1,
             'nthread' = 16)

# begin the training step
bst = xgboost(params=param, data=xgmat, nround=120)

# have already read in the test data, need to perform the next steps
data = as.matrix(test[2:31])
xgmat = xgb.DMatrix(data, missing=-999.0)

# make predictions for the test set
ypred = predict(bst, xgmat)

# recode the data according to the Kaggle Challenge format
idx <- test[[1]]
rorder = rank(ypred, ties.method='first')
threshold <-  0.15
ntop = length(rorder) - as.integer(threshold*length(rorder))
plabel = ifelse(rorder > ntop, 's', 'b')
outdata = list('EventID' = idx,
               "RankOrder" = rorder,
               'Class' = plabel)
write.csv(outdata, file='submission.csv', quote=FALSE, row.names=FALSE)



