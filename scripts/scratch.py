
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pandarallel import pandarallel


# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_lg')

stopwords = nlp.Defaults.stop_words

pandarallel.initialize()

all_tweets["tweet_lemmas"] = all_tweets["tweet_content"].parallel_apply(preprocess)

train_X, test_X, train_y, test_y = train_test_split(all_tweets['tweet_lemmas'], pd.get_dummies(all_tweets['party'])["D"], test_size=0.3, random_state=42, stratify=all_tweets['party'])

# Generating ngrams
vectorizer = CountVectorizer(ngram_range = (1,3))
vectorizer = TfidfVectorizer(ngram_range = (1,3))
train_X = vectorizer.fit_transform(train_X)
test_X = vectorizer.transform(test_X)

# Fit classifier
clf = MultinomialNB()
clf.fit(train_X, train_y)

accuracy = clf.score(test_X, test_y)
print("The accuracy of the classifier on the test set is %.3f" % accuracy)

