from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import re
import string
import nltk
# import en_core_web_sm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize

# nlp = en_core_web_sm.load()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
Stopwords = set(stopwords.words('english'))


n_gram_range = (3, 3)
stop_words = "english"

def clean_text(text):
  if not isinstance(text,float):
    text = text.replace('User-Agent:','')
    text = re.sub(r'[0-9]+', ' ', text).strip()
    text = text.lower()
    # text = re.sub('\[.*\]','', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('\S*\d\S*\s*','', text).strip()
    return text.strip()
  else:
    return ''

lemmatizer = WordNetLemmatizer()

def lemmetize_text(text):
  text = ' '.join(lemmatizer.lemmatize(word) for word in word_tokenize(text) if not word in Stopwords)
  return text

def getKeyWords(text):
    text = clean_text(text)
    text = lemmetize_text(text)
    words = text.split()
    text = " ".join(sorted(set(words), key=words.index))
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
    all_candidates =count.get_feature_names_out()
    return all_candidates[:5]