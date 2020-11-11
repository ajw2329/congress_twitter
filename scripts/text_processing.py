import spacy

# Load the en_core_web_sm model
nlp = spacy.load('en_core_web_sm')


def lemmatize(text, nlp_model):

    doc = nlp_model(text)

    lemmas = [token.lemma_ for token in doc]

    return lemmas


a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]


def preprocess(text):
    # Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    
    return ' '.join(a_lemmas)


# Generate tokens and pos tags
pos = [(token.text, token.pos_) for token in doc]


# Returns number of proper nouns
def proper_nouns(text, model=nlp):
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    
    # Return number of proper nouns
    return pos.count("PROPN")


# Print all named entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)


def find_persons(text):
  # Create Doc object
  doc = nlp(text)
  
  # Identify the persons
  persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
  
  # Return persons
  return persons

