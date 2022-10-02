# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:59:38 2022

@author: ALVIN
"""
#%%
import streamlit as st  # 🎈 data web app development
# from streamlit import caching

st.set_page_config(
    page_title="NTRS Document Retrieval System",
    # page_icon="✅",
    page_icon=":shark:",
    layout="wide",
)

# %%
# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 2rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)       

st.markdown("<h3 style='text-align: left;'><strong>NASA Space Apps Challenge 2022 - </strong>Can AI Preserve our science legacy?</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>NTRS Document Retrieval System</h1>", unsafe_allow_html=True)

st.write("Welcome to the NASA Technical Reports Server (NTRS) Document Retrieval System.")



col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    nuclear_physics = st.checkbox('Nuclear Physics')
with col2:
    optics = st.checkbox('Optics')
with col3:
    eee = st.checkbox('Electronics and Electrical Engineering')
with col4:
    struc_mech = st.checkbox('Structural Mechanics')
with col5:
    geophysics = st.checkbox('Geophysics')
with col6:
    energy_prod_conv = st.checkbox('Energy Production and Conversion')

query = st.text_input("Please enter your search here... 👇")
srch_button = st.button("Search")
# update_corpus = st.button("Update corpus")


import os
import json
import PyPDF2
import pickle
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts

from PyPDF2 import PdfReader

import nltk
from nltk.probability import FreqDist


from gensim import models
from gensim import corpora
from itertools import compress
from gensim import similarities
from collections import defaultdict

from analytics import *
from retrieval_system import *
from get_title_and_abstract import *

#%%

@st.cache()
def get_nltk_resources():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    nltk.download('punkt')

    return None


get_nltk_resources()

@st.cache
def get_docs(database_folder):
    files = os.listdir(database_folder)
    docs_dict = dict()

    for file in files:
        filename = os.path.join(database_folder, file)
        with open(filename,'rb') as f:
            f = PyPDF2.PdfFileReader(f)
            content = ''
            for page in f.pages:
                content = content + ' ' + page.extractText()
        
        analytics = get_analytics(filename, database_folder)
        print("\n analytics\n", analytics)
        docs_dict[file] = {'content':content, 'analytics':analytics}

    return docs_dict

@st.cache
def clean_docs(docs):
    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())

    document_tracker = []
    texts = []
    for key in docs.keys():
        content = docs[key]['content']
        content = [word for word in content.lower().split() if word not in stoplist]
        texts.append(content)
        document_tracker.append(key)

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    np.save('document_tracker.npy', np.array(document_tracker, dtype=object))
    return texts


@st.cache
def create_model(texts, num_topics=5):
    dictionary = corpora.Dictionary(texts)
    with open('dictionary.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle)

    corpus = [dictionary.doc2bow(text) for text in texts]
    np_corpus = np.array(corpus, dtype=object)
    np.save('corpus.npy', corpus)
    
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
    lsi.save('lsi.model')
    return


def read_pdfs(n):
    ''' This function reads a document given it's id.'''
    doc_folder = 'database'
    for i in range(n):
        doc_name = 'paper_'+str(i)+'.pdf'
        st.subheader(doc_name)
        doc_path = os.path.join(doc_folder, doc_name)
        print(doc_folder)
        print(doc_path)
        temp = open(doc_path, 'rb')
        PDF_read = PdfReader(temp)
        first_page = PDF_read.getPage(0)
        document = first_page.extractText()
        # document = 'My       name     is     alvin'
        # document = ' '.join(document.split())
        # print(document)
        

def get_title(doc_id, dataset_folder='database'):
    paper_path = os.path.join('database', doc_id)
    reader = PdfReader(paper_path)
    page = reader.pages[0]
    text = page.extract_text()
    text = text.split('\n')
    title = text[0]
    # st.subheader(title)
    return title


def get_ttle_n_abs(doc_id, dataset_folder='database'):
    pdf_file = os.path.join(dataset_folder, doc_id)
    # print("Pdf file: ", pdf_file)
    pdf_minr = get_text(pdf_file)
    abstract = journal_abs(pdf_minr)
    # print("Abstract:::", abstract)
    title = get_title(doc_id)
    # print("Title:::", title)
    return {'title': title, 'abstract':abstract}


def display_doc(doc, score):
    '''This function displays the document title, summary and keywords in the web app.'''
    st.header(doc['title'])
    st.write(score)
    st.subheader('Abstract')
    st.write(doc['abstract'])

    return None


dataset_folder = 'database'
subject_categories = ['Nuclear Physics',
                    'Optics',
                    'Electronics and Electrical Engineering',
                    'Structural Mechanics',
                    'Geophysics',
                    'Energy Production and Conversion']



selected_topics = [nuclear_physics, optics, eee, struc_mech,geophysics,energy_prod_conv]
selected_topics = list(compress(subject_categories, selected_topics))
corpus_folder = 'corpus'
if len(selected_topics) > 0:
    file_ids = os.listdir(corpus_folder)
    file_ids = set(map(lambda x: x.split('.')[0], file_ids))
    for file in file_ids:
        filename = os.path.join(corpus_folder, file+'.json')
        with open(filename) as f:
            file_details = json.load(f)
        
        if file_details['subjectCategory'] in selected_topics:
            # st.write(file_details)
            try:
                st.header(file_details['title'])
                st.subheader('Abstract')
                st.write(file_details['abstract'])
                html_str = f"""
                        <p><strong>Keywords: </strong>{file_details['keywords']}</p>
                        """
                st.markdown(html_str, unsafe_allow_html=True)
            except:
                pass


# if update_corpus:
#     docs = get_docs(dataset_folder)
#     with open('document_dictionary.pickle', 'wb') as handle:
#         pickle.dump(docs, handle)
    
#     texts = clean_docs(docs)

#     num_topics = 5
#     create_model(texts, num_topics=num_topics)


lsi = models.LsiModel.load("lsi.model")
corpus = list(np.load('corpus.npy', allow_pickle=True))
with open('dictionary.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)


doc_tracker = list(np.load('document_tracker.npy', allow_pickle=True))
with open('document_dictionary.pickle', 'rb') as handle:
    document_dictionary = pickle.load(handle)

# st.write(docs)
# st.write(texts)

if srch_button or query:
    st.write("Search Results")
    # st.write("Document:          Score")
    sims,topics = search_docs(query, lsi, dictionary, corpus)
    

    top_topics = []
    for topic in topics[:3]:
        top_topics.append(round(topic[0]))

    for sim in sims[:5]:
        doc = doc_tracker[sim[0]]
        print("\ndoc: ", doc)
        print(document_dictionary[doc]['analytics'])
        score = round(sim[1],3)
        title_n_abstract = get_ttle_n_abs(doc)
        display_doc(title_n_abstract, score)
        keywords = document_dictionary[doc]['analytics']
        keywords = [keyword[0] for keyword in keywords]
        keywords = ', '.join(keywords)
        
        col1,col2 = st.columns(2)
        with col1:
            html_str = f"""
                    <p><strong>Keywords: </strong>{keywords}</p>
                    """
            st.markdown(html_str, unsafe_allow_html=True)

        # with col2:
        #     html_str = f"""
        #             <p><strong>Topics: </strong>{top_topics}</p>
        #             """
        #     st.markdown(html_str, unsafe_allow_html=True)

    # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'


    # st.write("Searched for: " + query)

    # read_pdfs(5)
    # get_ttle_n_abs(7)
    # get_title(7)
    # title_n_abstract = [get_ttle_n_abs(7)]*2

    # display_doc(title_n_abstract)
    

st.markdown("<hr><p style='text-align:center; margin-top:3em;'>NTRS Document Retrieval System <br>Created by: <i color:#021691>256_Datanauts</i></p>", unsafe_allow_html=True)