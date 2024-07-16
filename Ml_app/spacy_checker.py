import spacy

def test_spacy():
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    # Process a sample text
    doc = nlp("SpaCy is a great library for natural language processing.")
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.is_stop)

test_spacy()
