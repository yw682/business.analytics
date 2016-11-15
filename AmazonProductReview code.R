library(quanteda)
library(stm)
library(tm)
library(NLP)
library(ggplot2)
library(ggdendro)
library(fpc)
library(dplyr)
library(stringr)
library(lda)
library(LDAvis)
library(dplyr)
require(magrittr)
library(openNLP)
library(servr)

#load .csv file with new articles
review <- read.csv("~/Desktop/Reviews.csv",
                   header = T, stringsAsFactors = F)

#passing Full Text to variable review.A
review.A <- review$Text

#Cleaning corpus
stop_words <- stopwords("SMART")
# additional junk words showing up in the data
stop_words <- c(stop_words, "I", "have", "a", "are", "is", "my","this", 
                "was", "for", "the", "these")
stop_words <- tolower(stop_words)

review.A <- gsub("'", "", review.A)
review.A <- gsub("[[:punct:]]", " ", review.A)
review.A <- gsub("[[:cntrl:]]", " ", review.A)  
review.A <- gsub("^[[:space:]]+", "", review.A)
review.A <- gsub("[[:space:]]+$", "", review.A) 
review.A <- gsub("[^a-zA-Z -]", " ", review.A) 
review.A <- tolower(review.A)

# get rid of blank docs
review.A <- review.A[review.A != ""]

# tokenize on space and output as a list:
list <- strsplit(review.A, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(list))
term.table <- sort(term.table, decreasing = T)

# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(list, get.terms)

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (1741)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (56196)
term.frequency <- as.integer(term.table) 

# MCMC and model tuning parameters:
K <- 10
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fit the model:
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1 

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

product.reviews <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

# create the JSON object to feed the visualization:
json <- createJSON(phi = product.reviews$phi, 
                   theta = product.reviews$theta, 
                   doc.length = product.reviews$doc.length, 
                   vocab = product.reviews$vocab, 
                   term.frequency = product.reviews$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)
