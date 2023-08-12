from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string # "string" module is already installed with Python

def clean_text(text):
    # Punctuation
    for punctuation in string.punctuation:
        text = text.replace(punctuation,'')

    # Lowercase
    text_ls = text.split()
    text_ls = [t.lower() for t in text_ls]
    text = ' '.join(text_ls)

    # Stop words
    for stop_word in set(stopwords.words('english')) :
        text.replace(stop_word,'')
    text_ls = text.split()

    # Numbers
    text_ls = [t if t.isalpha() else '__numeric__' for t in text_ls]

    # Lemmatizing
    # Verbs
    text_ls = [
        WordNetLemmatizer().lemmatize(t, pos = "v") # v --> verbs
        for t in text_ls
    ]

    # Nouns
    text_ls = [
        WordNetLemmatizer().lemmatize(t, pos = "n") # n --> nouns
        for t in text_ls # We used the list with lemmatized verbs
    ]

    return ' '.join(text_ls)
