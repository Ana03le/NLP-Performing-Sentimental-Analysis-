#final experiment
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
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=50)
len(unigram_feats)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)


bigram_feats = sentim_analyzer.bigram_collocation_feats(all_words_neg, min_freq=1000)
len(bigram_feats)
sentim_analyzer.add_feat_extractor(extract_bigram_feats,bigrams = bigram_feats)

all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(4000)
word_features = [word for (word, freq) in word_items]
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
sentim_analyzer.add_feat_extractor(document_features)



import pickle
f = open('my_classifier.pickle','wb')
pickle.dump(classifier, f)
f.close()

#load classifier
import pickle
f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

#testing with new documents
text = """sucks sucks sucks sucks sucks sucks """
tokens= nltk.regexp_tokenize(text, r'[?!"\']|[a-zA-Z]+[\'.][a-zA-Z]+|[a-zA-Z]+')
l_tokens = [ x.lower() for x in tokens]
newText = sentim_analyzer.extract_features(l_tokens)
classifier.classify(newText)