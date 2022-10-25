import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import contractions as ct
import re

stop_words = set(stopwords.words('english'))

def stopwordremoval(sent, stop_words, cnt =0):
  wordsFiltered = []
  words = word_tokenize(sent)
  for w in words:  
    if w not in stop_words:
      wordsFiltered.append(w)
  return wordsFiltered

def inf_clean(inf_sent: list):
    cleaned_data = []
    for sent in inf_sent:
        sent = " ".join(stopwordremoval(sent, stop_words))
        sent = sent.translate(str.maketrans("", "", string.punctuation))
        sent = sent.lower()
        sent = sent.replace('\n', '')
        sent = ct.fix(sent)
        sent = re.sub(r'[^a-zA-Z ]+', '', sent)
        sent = " ".join(sent.split())
        cleaned_data.append(sent)
        #print(cleaned_data)
        #print('cleaning completed successfully')
    return(cleaned_data)
