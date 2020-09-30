#####################
# EDA for Fake News #
#####################

# Libaries
library(tidyverse)
library(DataExplorer)
library(tidytext)
library(stringr)


# Load in data
train <- read_csv('train.csv')
test <- read_csv('test.csv')

# Combine test and training data
news <- bind_rows(train=train, test=test, .id="Set")

# Missing values?
plot_missing(train)
plot_missing(test)


# Tokenization
t <- tibble(id = news$id[1:20], text = news$text[1:20])
token <- t %>% unnest_tokens(word, text)
