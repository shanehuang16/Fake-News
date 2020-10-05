# Prepare workspace
rm(list=ls())
setwd("C:/Users/Arathen/Desktop/Github Projects/Fake-News/Eirik's Work")

# Load libraries
library(tidyverse)
library(wactor)
library(xgboost)

# Read data
news <- read_csv("./cleanedNews.csv")

# Split into training and test sets
train <- news %>% filter(Set == "train") %>% select(-Set)
test <- news %>% filter(Set == "test") %>% select(-Set)
dim(train)
dim(test)

# Create validation set from train
data <- split_test_train(train, .p = 0.85)

# Create wactor environment object
acc_v <- wactor(data$train$text, max_words = 10000)

# Create tf-idf matrices
data.train <- tfidf(acc_v, data$train$text)
data.test  <- tfidf(acc_v, data$test$text)

# Convert to DMatrix objects
xgb.data <- list(train = xgb_mat(data.train, y = data$train$label),
                 test = xgb_mat(data.test, y = data$test$label))

#####################
## Build XGB Model ##
#####################

params <- list(max_depth = 2,
               eta = 0.4,
               nthread = 60,
               objective = 'binary:logistic'
)

xgb.model <- xgb.train(params,
                        xgb.data$train,
                        nrounds = 100,
                        print_every_n = 50,
                        watchlist = xgb.data
)

# Increase nrounds and train on same model
xgb.model <- xgb.train(params,
                        xgb.data$train,
                        xgb_model = xgb.model,
                        nrounds = 500,
                        print_every_n = 50,
                        watchlist = xgb.data
)

formatC(head(predict(xgb.model, xgb.data$test)), format = 'f', digits = 10)

# Looking at a basic summary of the predicted probabilities:
summary(predict(xgb.model, xgb.data$test))

# Confusion Matrix for Evaluation
confusion <- table(predict = predict(xgb.model, xgb.data$test) >= 0.5,
                   actual = data$test$label)
confusion

# Calculating the Sensitivity:
round(confusion[2, 2]/(confusion[2, 2] + confusion[1, 2]), digits = 5)
# Calculating the Specificity:
round(confusion[1, 1]/(confusion[1, 1] + confusion[2, 1]), digits = 5)
# Calculating the Positive Predictive Value (PPV):
round(confusion[2, 2]/(confusion[2, 2] + confusion[2, 1]), digits = 5)
# Calculating the Negative Predictive Value (NPV):
round(confusion[1, 1]/(confusion[1, 1] + confusion[1, 2]), digits = 5)

# For actual predictions, prepare the test dataset
test.data <- tfidf(acc_v, test$text)
# Convert to DMatrix
test.xgb.data <- xgb_mat(test.data)
# Predict response using model
test$label <- as.integer(predict(xgb.model, test.xgb.data) >= 0.5)
# Create submission file
write_csv(select(test, id, label), 'ESSubMk1.csv')
