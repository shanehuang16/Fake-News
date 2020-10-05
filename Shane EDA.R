#####################
# EDA for Fake News #
#####################

# Libaries
library(tidyverse)
library(DataExplorer)
library(tidytext)
library(stringr)
library(SnowballC)
library(kableExtra)


# Load in data
train <- read_csv('train.csv')
test <- read_csv('test.csv')

# Combine test and training data
news <- bind_rows(train=train, test=test, .id="Set")

# Missing values?
plot_missing(train)
plot_missing(test)


# Tokenization
t <- tibble(id = news$id, text = news$text)
text_df <- t %>% unnest_tokens(word, text)

text_df$word <- wordStem(text_df$word,  language = "english")

head(table(text_df$word)) %>%
  kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)
