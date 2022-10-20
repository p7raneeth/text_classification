import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


stop_words = set(stopwords.words('english'))

def stopwordremoval(sent, stop_words, cnt =0):
  wordsFiltered = []
  words = word_tokenize(sent)
  for w in words:  
    if w not in stop_words:
      wordsFiltered.append(w)
  return wordsFiltered