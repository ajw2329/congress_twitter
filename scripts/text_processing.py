import spacy
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from pandarallel import pandarallel


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



class PartsOfSpeechExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self):
        pass

    def get_pos(self, text, model):
        """Helper code to compute average word length of a name"""
        doc = nlp(text)
        return Counter([token.pos_ for token in doc])

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        pandarallel.initialize()
        return pd.DataFrame(df['tweet_content'].parallel_apply(self.get_pos).to_list()).fillna(0)

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self