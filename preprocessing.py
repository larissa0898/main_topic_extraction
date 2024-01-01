from lxml import etree
from gensim.utils import tokenize
from nltk.corpus import stopwords
import spacy
import json
import re
import os
from wiki_dump_reader import Cleaner, iterate



def extracting_titles_and_texts(filename):
    """ A function that extracts relevant titles and texts from the Wiki-XML file to preprocess 
    it further and saving it to a dict. 

    Parameters
    ----------
    filename : str
        Name of the file from which the titles and texts are to be extracted.

    Returns
    -------
    wiki_dic : dict
        Contains the extracted titles (keys) and texts (values) of Wikipedia articles.
    """
    data_folder = "data"  # Name des Ordners, in dem die Dateien liegen

    # Pfad zur Datei im data-Ordner
    file_path = os.path.join(data_folder, filename)

    wiki_titles = []
    wiki_texts = []

    cleaner = Cleaner()
    for title, text in iterate(file_path):
        wiki_titles.append(title.lower())
        text = cleaner.clean_text(text)
        cleaned_text, _ = cleaner.build_links(text)
        wiki_texts.append(cleaned_text)


    wiki_dic = dict(zip(wiki_titles, wiki_texts))

    return wiki_dic




def regex_for_text_smoothing(wiki_dic):
    """ A function using regex for smoothing the texts to get rid
    of brackets, etc.

    Parameters
    ----------
    wiki_dic : dict
        Contains the extracted titles (key) and texts (value) of Wikipedia articles.
    
    Returns
    -------
    wiki_dic : dict
        Contains the smoothed titles (key) and texts (value) of Wikipedia articles without brackets, etc.
    """

    regex = r'[^\w\s]'


    wiki_titles = []
    wiki_texts = [] 

    pattern = re.compile(regex)
    for title in wiki_dic:
        wiki_texts.append(re.sub(pattern, '', wiki_dic[title]))
        wiki_titles.append(re.sub(pattern, ' ', title))
        
    wiki_dic = dict(zip(wiki_titles, wiki_texts))

    return wiki_dic




def tok_lemmatizing(wiki_dic, title):
    """ A function that tokenizes and lemmatizes the data. 

    Parameters
    ----------
    wiki_dic : dict
        Contains the smoothed titles (key) and texts (value) of Wikipedia articles without brackets, etc.
    title : str
        Contains a title of the list of Wikipedia topics.

    Returns
    -------
    data_lemma : list
        Contains the Wikipedia texts in tokenized and lemmatized form.
    """
    data = list(tokenize(wiki_dic[title]))  

    nlp = spacy.load('de_core_news_md')
    data_lemma = []

    for word in data:
        if len(word) > 3:
           doc = nlp(word)
           result = ' '.join([x.lemma_ for x in doc]) 
           data_lemma.append(result.lower())

    return data_lemma



def remove_stopwords(data_lemma, freq_based=True):
    """ A function that removes stopwords from data.

    Parameters
    ----------
    data_lemma : list
        Contains the Wikipedia texts in tokenized and lemmatized form.
    freq_based : bool
        If freq_based is true, the part in the if-condition is used for the frequency-based extraction.
        If freq_based is false, the else-part is used for the tf-idf extraction.

    Returns
    -------
    new_data : list
        Contains a list of words of a Wikipedia text without stopwords (frequency-based extraction).
    text : str
        Contains a Wikipedia text without stopwords (tf-idf extraction). 
    """
    german_stop_words = stopwords.words('german')
 
    if freq_based == True:
        new_data=[]

        for word in data_lemma:
            if word not in german_stop_words:
                new_data.append(word)
        return new_data

    else:
        text = [word for word in data_lemma if not word in  
                german_stop_words] 
        text = " ".join(text)
        return text



def save_corpus_in_json():
    """ A function that stores the data (preprocessed titles and texts) for the tf-idf method 
    in a json file to access it for tf-idf extraction.

    Parameters
    ----------
    -

    Returns
    -------
    None
    """
    texts = []
    titles = []
    for name in filenames:
        wiki_dic = extracting_titles_and_texts(name)
        wiki_dic = regex_for_text_smoothing(wiki_dic)
        for title in wiki_dic:
            titles.append(title)
            texts.append(remove_stopwords(tok_lemmatizing(wiki_dic, title), freq_based=False))
    
    corpus = dict(zip(titles, texts))

    with open('tf_idf_extraction.json', 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)




filenames = ["WikipediaZeichnen.xml", "WikipediaSportarten.xml", "WikipediaKlettern.xml"]