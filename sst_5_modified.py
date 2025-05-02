import spacy
import benepar
from nltk import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# Load dataset
dataset = pd.read_csv('sst_train.txt', sep='\t', header=None, names=["label", "text"])
dataset['label'] = dataset['label'].str.replace('__label__', '')
dataset['label'] = dataset['label'].astype(int)  # Categorical data type for truth labels
dataset['label'] = dataset['label'] - 1  # Zero-index labels for PyTorch

# Split into features and labels
sentences = dataset['text']
labels = dataset['label']

# Split into training and test sets (e.g., 80% train, 20% test)
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42, stratify=y  # stratify to preserve label distribution
)

# Convert text sentences to number form using tfidf vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',  # Use scikit-learn's English stop words
    lowercase=True
)

train_tfidf_matrix = vectorizer.fit_transform(train_sentences)
test_tfidf_matrix = vectorizer.transform(test_sentences)
# Define classifier and fit training data
classifier = LogisticRegression(C=10, penalty='l1', solver='saga', max_iter=10000)
classifier.fit(train_tfidf_matrix, train_labels)

y_pred = classifier.predict(test_tfidf_matrix)
accuracy = accuracy_score(test_labels, y_pred)
print(f"Training accuracy: {accuracy}")

def print_accuracy():
    return accuracy

# Load Spacy model and Benepar once at the start
nlp = spacy.load("en_core_web_md")
benepar.download('benepar_en3')
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

# Function to extract meaningful phrases from a sentence
def phrases_from_sentence(sentence):
    doc = nlp(sentence)  # Process the sentence with spacy and benepar
    phrases = []
    for sent in doc.sents:
        tree = sent._.parse_string  # Get the parse tree for the sentence
        parsed = Tree.fromstring(tree)  # Parse the tree using NLTK
        # Extract all phrases where the subtree height is greater than 2
        phrases += [' '.join(leaf) for subtree in parsed.subtrees() if subtree.height() > 2 for leaf in [subtree.leaves()]]
    return phrases

# Function to predict sentiment for a phrase
def predict_sentiment_phrase(phrase):
    test = vectorizer.transform([phrase])
    pred = classifier.predict(test)
    return pred[0]

# Function to predict sentiment for the sentence by analyzing its phrases
def predict_sentiment_sentence(sentence):
    phrases = phrases_from_sentence(sentence)
    len_phrases = len(phrases)
    value = 0
    for i in range(len_phrases):
        value += predict_sentiment_phrase(phrases[i])
    value /= len_phrases  # Compute average sentiment score based on phrases

    if value < 1.5:
        out = "This review is negative"
    elif value < 2.5:
        out = "This review is neutral"
    else:
        out = "This review is positive"
    
    return out
