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
training_docs = documents[:16]
testing_docs = documents[16:20]
def word_feats(words):
    return dict([(word, True) for word in words])
from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(word_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

#Tree Decision
from nltk.classify import DecisionTreeClassifier
trainer = DecisionTreeClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))


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
training_docs = documents[:16]
testing_docs = documents[16:20]
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
 
def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])
from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(stopword_filtered_word_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
#Tree Decision
from nltk.classify import DecisionTreeClassifier
trainer = DecisionTreeClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))
	
#punctuationFilter_feats
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
training_docs = documents[:16]
testing_docs = documents[16:20]
import string
def punctuationFilter_feats(words):
     return {word:True for word in words if word not in string.punctuation}

from nltk.sentiment import SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()
sentim_analyzer.add_feat_extractor(punctuationFilter_feats)
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
#Tree Decision
from nltk.classify import DecisionTreeClassifier
trainer = DecisionTreeClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)
print(nltk.classify.accuracy(classifier, test_set))
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))