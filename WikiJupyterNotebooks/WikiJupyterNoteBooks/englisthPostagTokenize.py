from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def get_token(text):
    return [x[0] for x in pos_tag(word_tokenize(text))]

def get_tag(text):
    return [x[1] for x in pos_tag(word_tokenize(text))]