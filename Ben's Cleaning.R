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
library(textcat)
library(xml2)
library(rvest)
library(stopwords)
library(stringr)

# Load in data
train <- read_csv('train.csv')
test <- read_csv('test.csv')

# Combine test and training data
news <- bind_rows(train=train, test=test, .id="Set")

news['language'] <- textcat(news$text)

news$language %>% table() %>% sort()
languages <- news$language %>% unique() %>% na.omit()

# Find and remove stopwords for each of those languagues. 

codes <- read_html("https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes")
code.df <- codes %>% html_table(fill = TRUE) %>% .[[2]] %>% .[, c(3, 5)]
code.df[, 1] <- code.df[, 1] %>% tolower()

able <- sapply(languages, 
               function(x) sum(grepl(substr(x, 1, 4), code.df[, 1])) %>% as.logical()) %>% .[which(. == TRUE)] %>% names()



current.code <- substr(news$language[1], 1, 4) %>% grepl(code.df[, 1]) %>% which() %>% .[1] %>% code.df$`639-1`[.]
no.stopwords <- mapply(function(i) {sapply(str_split(i, " "), 
                                           function(x) !(tolower(x) %in% stopwords(language = current.code))) %>% str_split(news$text[1], " ")[[1]][.]}, i = news$text)

      