# Word Frequency
# Quote Number
# Dates
# Stopwords
# Length

# There are different languages
# Take out stopwords for each language

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

library(textcat)

textcat(train$text[1:5]) %>% unique
