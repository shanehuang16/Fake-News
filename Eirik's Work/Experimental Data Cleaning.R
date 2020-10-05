# Prepare workspace
rm(list=ls())
setwd("C:/Users/Arathen/Desktop/Github Projects/Fake-News/Eirik's Work")

# Load libraries
library(readr)
library(dplyr)
library(cld2)
library(stringr)
library(ggplot2)
library(tidyr)
library(tm)       ## For text mining
library(textstem)     ## For lemmatization
library(tidytext)
library(wordcloud2)
library(pROC)
library(ROCR)
library(caret)
library(DataExplorer)
library(tidyverse)
library(wactor)
library(xgboost)

# Read in data
train <- read_csv("../train.csv")
test <- read_csv("../test.csv")
news <- bind_rows(train=train, test=test, .id="Set")

# Fuse title with text
news$text <- with(news, paste(title, text))
# Analyze languages
news$lan <- cld2::detect_language(text = news$text, plain_text = FALSE)
# Group languages
for (i in 1:length(news$lan)) {
  if ((is.na(news$lan[i]) == TRUE) | (news$lan[i] %in% c('ar','ca','el','id','it',
                                                          'nl','no','pl','pt','sr',
                                                          'tr','zh'))) {
    news$lan[i] = 'other'
  }
}
# Count the number of quotation marks in the text, which could indicate legitimacy.
for (i in 1:length(news$text)) {
  x <- news$text[i]
  news$quotenum[i] <- lengths(regmatches(x, gregexpr("\"", x)))
}
# Count length of title
for (i in 1:length(news$title)) {
  if (is.na(news$title[i]) == TRUE) {
    news$title[i] = 0
  } else {
    news$title[i] = str_length(news$title[i])
  }
  x <- news$text[i]
  news$quotenum[i] <- lengths(regmatches(x, gregexpr("\"", x)))
}

plot_missing(news)

# Create VCorpus for cleaning
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

# Create Tidy data
df.tidy <- tidy(doc)

# Convert dtm to matrix
data.mat <- as.matrix(df.tidy)
dim(data.mat)

# Convert matrix to data frame
news.clean <- as.data.frame(data.mat)
# Replace news$text with cleaned data
news$text <- news.clean$text

write_csv(x=news, path="./cleanedNews.csv")
