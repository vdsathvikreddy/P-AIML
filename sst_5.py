from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import spacy
import benepar
from nltk import Tree

# load datasets separating with \t into two coloumns
dataset = pd.read_csv('sst_train.txt', sep = '\t', header = None, names = ["label", "text"])
dataset['label'] = dataset['label'].str.replace('__label__','')
dataset['label'] = dataset['label'].astype(int)   # Categorical data type for truth labels
dataset['label'] = dataset['label'] - 1  # Zero-index labels for PyTorch

# initialize sentences and labels to train
train_sentences = dataset['text']
train_labels = dataset['label']

# convert text sentences to number form using tfidf vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',  # Use scikit-learn's English stop words
    lowercase=True
)

train_tfidf_matrix = vectorizer.fit_transform(train_sentences)

# define classifier and fit training data
classifier = LogisticRegression(C=10, penalty='l1', solver='saga', max_iter=10000)

classifier.fit(train_tfidf_matrix, train_labels)

y_pred = classifier.predict(train_tfidf_matrix)
accuracy = accuracy_score(train_labels, y_pred)

print(accuracy)

sentence = 'amazing movie but graphics are the best'
s = [sentence]
test = vectorizer.transform(s)

print(classifier.predict(test))

def phrases_from_sentence(sentence):
  nlp = spacy.load("en_core_web_md")
# if not benepar.is_loaded('benepar_en3'):
  benepar.download('benepar_en3')
  nlp.add_pipe("benepar", config={"model": "benepar_en3"})
  
  text = "The movie was surprisingly delightful and beautifully directed."
  doc = nlp(text)
  
  for sent in doc.sents:
      tree = sent._.parse_string
  
  parsed = Tree.fromstring(tree)
  phrases = [' '.join(leaf) for subtree in parsed.subtrees() if subtree.height() > 2 for leaf in [subtree.leaves()]]
  return phrases
  
def predict_sentiment_phrase(phrase):
    s= [phrase]
    test = vectorizer.transform(s)
    pred = classifier.predict(test)
    return pred[0]

def predict_sentiment_sentence(sentence):
  phrases = phrases_from_sentiment(sentence)
  len = len(phrases)
  value = 0
  for i in range(1, len):
    value += predict_sentiment_phrase(phrases[i])
  value /= len
  if(value == 2):
    out = "This review is neutral"
  elif value < 2:
    out = "This review is negative"
  else:
    out = "This review is positive"
  return out
