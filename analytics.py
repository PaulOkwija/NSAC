import os
import nltk
import PyPDF2
from nltk.corpus import stopwords
import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

def get_analytics(doc_id, databaset_folder):
	filename = os.path.join(databaset_folder, doc_id)
	with open(filename,'rb') as f:
	    f = PyPDF2.PdfFileReader(f)
	    content = ''
	    for page in f.pages:
	        content = content + ' ' + page.extractText()
	

	lemmatizer = WordNetLemmatizer()
	tokens = [lemmatizer.lemmatize(token) for token in content]
	
	#Remove Stopwords and Punctuations
	punctuation = ['(',')',';',':','[',']',',']
	stop_words = stopwords.words('english')
	keywords = [word for word in tokens if not word in stop_words and  not word in punctuation and word.isalpha()]

	fqdist = FreqDist(keywords)
	freqdist = nltk.FreqDist(keywords)
	# print("\n")
	# print(freqdist)
	# print("\n")
	return list(freqdist.most_common(4))