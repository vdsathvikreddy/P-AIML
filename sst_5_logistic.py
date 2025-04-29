from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd

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

def predict_sentiment(sentence):
    s= [sentence]
    test = vectorizer.transform(s)
    pred = classifier.predict(test)
    if pred == [0]:
        out = "very positive!!"
    elif pred == [1]:
        out = "positive!"
    elif pred == [2]:
        out = "neutral :)"
    elif pred == [3]:
        out = "negative!"
    elif pred == [4]:
        out = "very negative!!"
    return out
