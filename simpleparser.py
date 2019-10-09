import nltk as nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def stemSentence(sentence):
    stop_words = set(stopwords.words('english'))
    englishStemmer=SnowballStemmer("english")
    
    tokenizer = RegexpTokenizer(r'\w+')
    token_words = tokenizer.tokenize(sentence)

    stem_sentence=[]
    for word in token_words:
        stem = englishStemmer.stem(word)
        if stem not in stop_words: 
            stem_sentence.append(stem)
            stem_sentence.append(" ")
            
    return "".join(stem_sentence)


def stemCorpus(corpus):
    newCorpus = []
    for sentence in corpus:
        stem_sentence = stemSentence(sentence)
        newCorpus.append(stem_sentence)
        
    return newCorpus

