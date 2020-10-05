# Prepare workspace
rm(list=ls())
setwd("C:/Users/Arathen/Desktop/Github Projects/Fake-News/Eirik's Work")

# Initialize Libraries
library(readr)
library(dplyr)
library(stringr)
library(ggplot2)
library(tidyr)
library(tm)       ## For text mining
library(textstem)     ## For lemmatization
library(tidytext)
library(wordcloud2)
library(pROC)
library(ROCR)
library(randomForest)   ## Random forest classification
library(naivebayes)
library(caret)
library(DataExplorer)

# Read in data (Make sure to use UTF-8 encoding)
train <- read.csv("../train.csv", encoding="UTF-8", header=TRUE)
test <- read.csv("../test.csv", encoding="UTF-8", header=TRUE)
news <- bind_rows(train=train, test=test, .id="Set")

glimpse(news)

#########
## EDA ##
#########

plot_missing(news %>% select(-label))
hist(sapply(news$text, str_length))
news %>%
  count(author, sort=TRUE) %>%
  dim() # Looks like there are 4846 unique authors
news %>%
  count(author, sort=TRUE) %>%
  filter(n > 10) #About 500 authors wrote more than 10 articles

###########################################
## Data Cleaning and Feature Engineering ##
###########################################

# There are a lot of anonymous or unknown authors. This will try to group them all.
#anon_indices <- news %>% filter(author %in% c('staff writer', 'rt in español',
#                             'rt <U+043D><U+0430> <U+0440><U+0443><U+0441><U+0441><U+043A><U+043E><U+043C>',
#                             'no author', 'rt', 'author', 'guest', 'admin', 
#                             '-no author-', 'anonymous', 'editor', 'admin',
#                             'anonymous', 'nan', 'no author')) %>% select(author)
#news$author <- replace(news$author, unlist(anon_indices), c("Anonymous"))
#head(anon_indices)
#head(news$author, 20)

# Count the number of quotation marks in the text, which could indicate legitimacy.
for (i in 1:length(news$text)) {
  x <- news$text[i]
  news$quotenum[i] <- lengths(regmatches(x, gregexpr("\"", x)))
}

doc <- VCorpus(VectorSource(news$text))
# Convert text to lower case
doc <- tm_map(doc, content_transformer(tolower))

# Remove numbers
doc <- tm_map(doc, removeNumbers)

# Remove Punctuations
doc <- tm_map(doc, removePunctuation)

# Remove Stopwords
doc <- tm_map(doc, removeWords, stopwords('english'))

doc <- tm_map(doc, content_transformer(str_remove_all), "[[:punct:]]")

# Remove Whitespace
doc <- tm_map(doc, stripWhitespace)

# inspect output
writeLines(as.character(doc[[45]]))

# Lemmatization
doc <- tm_map(doc, content_transformer(lemmatize_strings))

# Create Document Term Matrix
dtm <- DocumentTermMatrix(doc)
inspect(dtm)

# remove all terms whose sparsity is greater than the threshold (x)
dtm.clean <- removeSparseTerms(dtm, sparse = 0.99)
inspect(dtm.clean)

# Create Tidy data
df.tidy <- tidy(dtm.clean)
df.word<- df.tidy %>% 
  select(-document) %>%
  group_by(term) %>%
  summarize(freq = sum(count)) %>%
  arrange(desc(freq))

# Word cloud
set.seed(1234) # for reproducibility 
wordcloud2(data=df.word, size=1.6, color='random-dark')

# Convert dtm to matrix
dtm.mat <- as.matrix(dtm.clean)
dim(dtm.mat)

# dtm.df <- as.data.frame(dtm.mat)
dtm.mat <- cbind(dtm.mat, Set = news$Set, Response = news$label)
dtm.mat[1:10, c(1, 2, 3, ncol(dtm.mat))]

# Convert matrix to data frame
dtm.df <- as.data.frame(dtm.mat)
dim(dtm.df)

# Split into train and test
train <- dtm.df %>% filter(Set == "train") %>% select(-Set)
test <- dtm.df %>% filter(Set == "test") %>% select(-Set)
dim(train)
dim(test)

# XGBoost Model
## Baseline model
grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

xgb_base <- train(form=Response~.,
                  data=train,
                  method="xgbTree",
                  trControl=train_control,
                  tuneGrid=grid_default,
                  verbose=TRUE
)

## Next, start tuning hyperparameters
nrounds <- 1000
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

xgb_tune <-train(form=imdb_score~.,
                 data=(imdb.train %>% select(-Set, -movie_title)),
                 method="xgbTree",
                 trControl=tune_control,
                 tuneGrid=tune_grid,
                 verbose=TRUE
)

# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)
xgb_tune$bestTune

## Next round of tuning
tune_grid2 <- expand.grid(nrounds = seq(from = 50, to = nrounds, by = 50),
                          eta = xgb_tune$bestTune$eta,
                          max_depth = ifelse(xgb_tune$bestTune$max_depth == 2,
                                             c(xgb_tune$bestTune$max_depth:4),
                                             xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
                          gamma = 0,
                          colsample_bytree = 1,
                          min_child_weight = c(1, 2, 3),
                          subsample = 1
)

xgb_tune2 <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid2,
  verbose=TRUE
)

tuneplot(xgb_tune2)
xgb_tune2$bestTune
min(xgb_tune$results$RMSE)
min(xgb_tune2$results$RMSE)

## Next tuning round
tune_grid3 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_tune3 <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid3,
  verbose=TRUE
)

tuneplot(xgb_tune3, probs = .95)
xgb_tune3$bestTune
min(xgb_tune$results$RMSE)
min(xgb_tune2$results$RMSE)
min(xgb_tune3$results$RMSE)

## Tuning the Gamma
tune_grid4 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune4 <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid4,
  verbose=TRUE
)

tuneplot(xgb_tune4)
xgb_tune4$bestTune
min(xgb_tune$results$RMSE)
min(xgb_tune2$results$RMSE)
min(xgb_tune3$results$RMSE)
min(xgb_tune4$results$RMSE)

## Reduce learning rate
tune_grid5 <- expand.grid(
  nrounds = seq(from = 100, to = 10000, by = 100),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = xgb_tune4$bestTune$gamma,
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune5 <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid5,
  verbose=TRUE
)

tuneplot(xgb_tune5)
xgb_tune5$bestTune
min(xgb_tune$results$RMSE)
min(xgb_tune2$results$RMSE)
min(xgb_tune3$results$RMSE)
min(xgb_tune4$results$RMSE)
min(xgb_tune5$results$RMSE)

## Fit the model and predict
final_grid <- expand.grid(
  nrounds = xgb_tune5$bestTune$nrounds,
  eta = xgb_tune5$bestTune$eta,
  max_depth = xgb_tune5$bestTune$max_depth,
  gamma = xgb_tune5$bestTune$gamma,
  colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
  min_child_weight = xgb_tune5$bestTune$min_child_weight,
  subsample = xgb_tune5$bestTune$subsample
)

xgb_model <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=final_grid,
  verbose=TRUE
)

xgb.preds <- data.frame(Id=imdb.test$movie_title, Predicted=predict(xgb_model, newdata=imdb.test))
head(xgb.preds, 25)
write_csv(x=xgb.preds, path="./XGBPredictions.csv")

