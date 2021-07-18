import spacy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from pandarallel import pandarallel
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



nlp = spacy.load('en_core_web_lg')

stopwords = nlp.Defaults.stop_words

def lemmatize(text, model = nlp):

    doc = model(text)

    lemmas = [token.lemma_ for token in doc]

    return lemmas


#a_lemmas = [lemma for lemma in lemmas 
#            if lemma.isalpha() and lemma not in stopwords]


def preprocess(text, model = nlp, stopwords = stopwords):
    # Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    
    return ' '.join(a_lemmas)


def get_pos(text, model = nlp):

    doc = model(text)

    pos = Counter([token.pos_ for token in doc])

    return pos


def get_sentiment(text, model = nlp):

    sentiment_analyzer = SentimentIntensityAnalyzer()

    return sentiment_analyzer.polarity_scores(text)



# Returns number of proper nouns
def proper_nouns(text, model=nlp):

    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of proper nouns
    return pos.count("PROPN")


# Print all named entities and their labels

def get_ents(text, model = nlp):

    doc = model(text)
    ents = doc.ents

    ents = [(ent.text, ent.label_) for ent in ents]

    return ents


def find_persons(text, model = nlp):
    # Create Doc object
    doc = model(text)
  
    # Identify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
  
    # Return persons
    return persons



def extract_test_handles():

    pass


def partition_handles_cv(nfold = 5):

    pass


class PartsOfSpeechExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def get_pos(self, text, model = nlp):
        """Helper code to compute average word length of a name"""
        doc = nlp(text)
        return Counter([token.pos_ for token in doc])

    def transform(self, series, y=None):
        """The workhorse of this feature extractor"""
        return pd.DataFrame(series.apply(self.get_pos).to_list()).fillna(0)

    def fit(self, series, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self



## below class from https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector(BaseEstimator, TransformerMixin):
    #Class Constructor 
    def __init__(self, feature_names):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit(self, X, y = None):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform(self, X, y = None):
        return X[ self._feature_names ] 

default_preprocessor = CountVectorizer().build_preprocessor()

def build_preprocessor(field):

    field_idx = list(train.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])


def main():

    twitter_handles = pd.read_csv("../data/congress_twitter_handles.csv")\
        .drop("Unnamed: 0", axis = "columns")\
        .rename(columns = {
            "Party": "party", 
            "Twitter": "username", 
            "Member of Congress": "chamber", 
            "Name": "name",
            "State":"state"})\
        .dropna(subset = ["username"])

    twitter_handles["username"] = twitter_handles["username"].apply(lambda x: x.replace("@", ""))
    twitter_handles["chamber"] = twitter_handles["chamber"].apply(lambda x: x.replace("U.S. ", ""))

    twitter_handles.to_csv("../data/congress_twitter_handles_cleaned.csv")

    usernames = set([i.replace("@", "") for i in twitter_handles.username])



    all_tweets = pd.read_csv("../output/all_tweets.csv")

    all_tweets = pd.merge(all_tweets, twitter_handles, how = "inner", on = ["username", "party"])


    all_tweets = all_tweets[all_tweets["username"].isin(usernames)]


    # add lemmas


    nlp = spacy.load('en_core_web_lg')

    stopwords = nlp.Defaults.stop_words

    pandarallel.initialize()

    all_tweets["tweet_lemmas"] = all_tweets["tweet_content"].parallel_apply(preprocess)


    # add pos

    pos_df = pd.DataFrame(all_tweets["tweet_content"].parallel_apply(get_pos).to_list()).fillna(0)

    #all_tweets = pd.concat([all_tweets, pos_df], axis = 1, ignore_index = True)

    all_tweets = pd.concat([all_tweets, pos_df], axis = 1)


    # add sentiment

    sentiment_df = pd.DataFrame(all_tweets["tweet_content"].parallel_apply(get_sentiment).to_list())

    all_tweets = pd.concat([all_tweets, sentiment_df], axis = 1)

    all_tweets.to_csv("../data/all_tweets_full.csv")


if __name__ == "__main__":

    main()