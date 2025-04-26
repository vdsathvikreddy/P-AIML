from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

data = load_dataset("sst2")

train_sentences = data['train']['sentence']

train_labels = data['train']['label']

test_sentences = data['test']['sentence']

test_labels = data['test']['label']

validation_sentences = data['validation']['sentence']

validation_labels = data['validation']['label']

vectorizer = TfidfVectorizer(
    stop_words='english',  # Use scikit-learn's English stop words
    lowercase=True
    ngram_range(2,2)
)

train_tfidf_matrix = vectorizer.fit_transform(train_sentences)
validation_tfidf_matrix = vectorizer.transform(validation_sentences)
test_tfidf_matrix = vectorizer.transform(test_sentences)

model = LogisticRegression(max_iter=10000)
model.fit(train_tfidf_matrix, train_labels)

# y_pred = model.predict(validation_tfidf_matrix)
# print(classification_report(validation_labels, y_pred))

# test_pred = model.predict(test_tfidf_matrix)

# def print_classification():
#     return classification_report(validation_labels, y_pred)
sentence = 'amazing movie but graphics are the best'
s = [sentence]
test = vectorizer.transform(s)

print(model.predict(test))

def predict_sentiment(sentence):
    s= [sentence]
    test = vectorizer.transform(s)
    pred = model.predict(test)
    if pred == [1]:
        out = "it is positive comment!! Thank you:)"
    else:
        out = "we will try to do ur best :("
    return out
