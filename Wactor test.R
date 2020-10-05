# Libaries
library(tidyverse)
library(DataExplorer)
library(tidytext)
library(stringr)
library(SnowballC)
library(kableExtra)
library(wactor)
library(xgboost)


# Load in data
train <- read_csv('train.csv')
test <- read_csv('test.csv')

# Pull out a sample of the training data for memory purposes
train <- train[sample(nrow(train), floor(nrow(train)*.50)),]

# Merge title and text
train$text <- with(train, paste(title, text))

# Split into train and test sets
news <- wactor::split_test_train(train, .p = 0.75)

# Create wactor
accuracy_v <- wactor::wactor(news$train$text,
  tokenizer = function(x) tokenizers::tokenize_words(x, strip_numeric = TRUE),
  max_words = 10000)

# Generate tfidf
news_train <- wactor::tfidf(accuracy_v, news$train$text)
news_test  <- wactor::tfidf(accuracy_v, news$test$text)

# split into test/train (and add opinion classifier scores)
news_xgb_data <- list(
  train = wactor::xgb_mat(news_train,
    y = news$train$label),
  test = wactor::xgb_mat(news_test,
    y = news$test$label)
)


# Model Fitting
param <- list(
    max_depth = 2,
    eta = 0.4,
    nthread = 60,
    objective = "binary:logistic"
)

fakenews_xgb_model <- xgboost::xgb.train(
    param,
    news_xgb_data$train,
    nrounds = 100,
    print_every_n = 50,
    watchlist = news_xgb_data
)

# Re-iterate
fakenews_xgb_model <- xgboost::xgb.train(
    param,
    news_xgb_data$train,
    xgb_model = fakenews_xgb_model,
    nrounds = 500,
    print_every_n = 50,
    watchlist = news_xgb_data
)
