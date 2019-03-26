import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
   for category in movie_reviews.categories()
   for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)
len(documents)
from nltk.sentiment.util import *
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
training_docs = documents[:1600]
testing_docs = documents[1600:2000]
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=10)
len(unigram_feats)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
from nltk.classify import NaiveBayesClassifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
     print('{0}: {1}'.format(key, value))
classifier.show_most_informative_features(10)

#Bigrams
import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
   for category in movie_reviews.categories()
   for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)
len(documents)
from nltk.sentiment.util import *
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
training_docs = documents[:1600]
testing_docs = documents[1600:2000]
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
bigram_feats = sentim_analyzer.bigram_collocation_feats(all_words_neg, min_freq=500)
len(bigram_feats)
sentim_analyzer.add_feat_extractor(extract_bigram_feats,bigrams = bigram_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
from nltk.classify import NaiveBayesClassifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
     print('{0}: {1}'.format(key, value))
classifier.show_most_informative_features(10)

#Most common words
import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
   for category in movie_reviews.categories()
   for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)
len(documents)
from nltk.sentiment.util import *
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
training_docs = documents[:1600]
testing_docs = documents[1600:2000]
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(5000)
word_features = [word for (word, freq) in word_items]
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(document_features)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

#Bayes
from nltk.classify import NaiveBayesClassifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

classifier.show_most_informative_features()

#subjectivity
import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
   for category in movie_reviews.categories()
   for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)
len(documents)
from nltk.sentiment.util import *
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
training_docs = documents[:1600]
testing_docs = documents[1600:2000]
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, freq) in word_items]
SLpath = 'subjclueslen1-HLTEMNLP05.tff'
def readSubjectivity(path):
   flexicon = open(path, 'r')
   sldict = { }
   for line in flexicon:
     fields = line.split()
     strength = fields[0].split("=")[1]
     word = fields[2].split("=")[1]
     posTag = fields[3].split("=")[1]
     stemmed = fields[4].split("=")[1]
     polarity = fields[5].split("=")[1]
     if (stemmed == 'y'):
       isStemmed = True
     else:
       isStemmed = False
     sldict[word] = [strength, posTag, isStemmed, polarity]
   return sldict

SL = readSubjectivity(SLpath)
def SL_features(document):
   document_words = set(document)
   features = {}
   for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
   weakPos = 0
   strongPos = 0
   weakNeg = 0
   strongNeg = 0
   for word in document_words:
      if word in SL:
        strength, posTag, isStemmed, polarity = SL[word]
        if strength == 'weaksubj' and polarity == 'positive':
           weakPos += 1
        if strength == 'strongsubj' and polarity == 'positive':
           strongPos += 1
        if strength == 'weaksubj' and polarity == 'negative':
           weakNeg += 1
        if strength == 'strongsubj' and polarity == 'negative':
           strongNeg += 1
        features['positivecount'] = weakPos + (2 * strongPos)
        features['negativecount'] = weakNeg + (2 * strongNeg)
   return features
from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(SL_features)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

#Bayes
from nltk.classify import NaiveBayesClassifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

classifier.show_most_informative_features()

#Negation features
import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
   for category in movie_reviews.categories()
   for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)
len(documents)
from nltk.sentiment.util import *
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
training_docs = documents[:1600]
testing_docs = documents[1600:2000]
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(10000)
word_features = [word for (word, freq) in word_items]
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
def NOT_features(document):
   features = {}
   for word in word_features:
     features['contains({})'.format(word)] = False
     features['contains(NOT{})'.format(word)] = False
   for i in range(0, len(document)):
     word = document[i]
     if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
        i += 1
        features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
     else:
        features['contains({})'.format(word)] = (word in word_features)
   return features

from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(NOT_features)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
from nltk.classify import NaiveBayesClassifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

classifier.show_most_informative_features()

#punctuation marks
import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
   for category in movie_reviews.categories()
   for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)
len(documents)
from nltk.sentiment.util import *
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
training_docs = documents[:1600]
testing_docs = documents[1600:2000]
import string
def punctuationFilter_feats(words):
     return {word:True for word in words if word in string.punctuation}

from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(punctuationFilter_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

#Bayes
from nltk.classify import NaiveBayesClassifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))
classifier.show_most_informative_features()

#stopwords
import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
   for category in movie_reviews.categories()
   for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)
len(documents)
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
training_docs = documents[:1600]
testing_docs = documents[1600:2000]
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
 
def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word in stopset])
from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(stopword_filtered_word_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

#Bayes
from nltk.classify import NaiveBayesClassifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))
classifier.show_most_informative_features()

#bag of words
import nltk
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
   for category in movie_reviews.categories()
   for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)
len(documents)
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
training_docs = documents[:1600]
testing_docs = documents[1600:2000]
def word_feats(words):
    return dict([(word, True) for word in words])
from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(word_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

#Bayes
from nltk.classify import NaiveBayesClassifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))
classifier.show_most_informative_features()