import nltk
#utf8
f = open('comment.txt', encoding = 'utf-8')
#read the document
text_wikipedia = f.read()
#load beautifulsoup package
from bs4 import BeautifulSoup
#parser the text (eliminate html elements)
raw = BeautifulSoup(text_wikipedia).get_text()
#make sentences
textsplit = nltk.sent_tokenize(raw)
#total of sentences
len(textsplit)
#tokenize the sentences
tokentext = [nltk.word_tokenize(sent) for sent in textsplit]
len(tokentext)
#make lower case all the tokens
sentences = []
for token in tokentext:
	sentence = []
	for tok in token:
	   sentence.append(tok.lower())
	sentences.append(sentence)
#I decided to use the function that we create in class for eliminating punctuaction marks
import re
def alpha_filter(w):
  pattern = re.compile('^[^a-z]+$')
    if(pattern.match(w)):
        return True
	else:
	    return False
sentences_alpha = []
for token in tokentext:
	sentence = []
	for tok in token:
		if not alpha_filter(tok):
			sentence.append(tok)
	sentences_alpha.append(sentence)
#I noticed that alpha_filter function was not eliminating all punctuaction marks, I will use the alpha function
sentences_alpha = []
for token in tokentext:
	sentence = []
	for tok in token:
		if(tok.isalpha()):
			sentence.append(tok)
	sentences_alpha.append(sentence)
len(sentences_alpha)
sentences_alpha[:4]

stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('a')
stopwords.append('the')
stopwords.append('href')
stopwords.append('class')
stopwords.append('title')
stopwords.append('wiki')
stopwords.append('wikipedia')
stopwords.append('amp')
stopwords.append('article')
stopwords.append('external')
stopwords.append('nofollow')
stopwords.append('sources')
stopwords.append('http')
stopwords.append('text')
stopwords.append('b')
stopwords.append('dd')
stopwords.append('span')
stopwords.append('user')
stopwords.append('page')
stopwords.append('br')
stopwords.append('p')
stopwords.append('li')
stopwords.append('hr')
stopwords.append('https')
stopwords.append('font')
stopwords.append('autonumber')
stopwords.append('autonumber')
stopwords.append('autosigned')
stopwords.append('unsigned')
stopwords.append('signatures')

stopwords_comments = []
for token in sentences_alpha:
	sentence = []
	for tok in token:
		if tok not in stopwords:
			sentence.append(tok)
	stopwords_comments.append(sentence)
len(stopwords_comments)
stopwords_comments[:4]
sentences = []
for sent in stopwords_comments:
    new_sent = ""
	for tok in sent:
	    new_sent = new_sent + tok + " "
	sentences.append(new_sent)
	

positive_sentences = []
negative_sentences = []
for sent in sentences:
  newText = sentim_analyzer.extract_features(sent)
  decision = classifier.classify(newText)
  if(decision == "pos"):
     positive_sentences.append(sent)
  else:
     negative_sentences.append(sent)
	 
	 
doc = word_tokenize(Example_Text.lower())

featurized_doc = {c:True for c in Example_Text.split()}
tagged_label = classifier.classify(featurized_doc)
print(tagged_label)


positive_sentences = []
negative_sentences = []
for sent in sentences:
    featurized_doc = {c:True for c in sent.split()}
    tagged_label = classifier.classify(featurized_doc)
    if(tagged_label == 'pos'):
       positive_sentences.append(sent)
    if(tagged_label == 'neg'):
       negative_sentences.append(sent)
	   
with open('negative_comments.txt', 'a') as f:
    for sent in negative_sentences:
	     sent = sent + '\n'
	     f.write(sent)