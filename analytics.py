import os
import nltk
import PyPDF2
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

def get_analytics(filename, databaset_folder):
	with open(filename,'rb') as f:
	    f = PyPDF2.PdfFileReader(f)
	    text = ''
	    for page in f.pages:
	        text = text + ' ' + page.extractText()
	
	tokens = word_tokenize(text)
	tokens = [w.lower() for w in tokens]
	
	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(token) for token in tokens]
	
	#Remove Stopwords and Punctuations
	punctuation = ['(',')',';',':','[',']',',']
	stop_words = stopwords.words('english')
	custom_stop_words = ['et', 'al', 'http', 'j']
	stop_words.extend(custom_stop_words)

	keywords = [word for word in tokens if not word in stop_words and  not word in punctuation and word.isalpha()]
	print("keywords: ", keywords)
	fqdist = FreqDist(keywords)
	freqdist = nltk.FreqDist(keywords)
	# print("\n")
	# print(freqdist)
	# print("\n")
	return list(freqdist.most_common(4))