import re
from lxml import etree
from gensim.utils import tokenize
import spacy
from nltk.corpus import stopwords
from collections import Counter
import heapq
import json



def extracting_titles_and_texts(filename):
    """ A function that extracts relevant titles and texts from the Wiki-XML file to preprocess 
    it further and saving it to a dict. 

    Parameters
    ----------
    filenames : str
        Name of file we want to extracts titles and texts.

    Returns
    -------
    wiki_dic : dict
        Contains the extracted titles (key) and texts (value) of Wikipedia articles.
    wiki_titles : list
        Contains list of titles of Wikipedia articles. 
    """
    tree = etree.parse(filename)
    root = tree.getroot()

    wiki_titles = []
    wiki_texts = []


    for title in root.findall(".//title", namespaces=root.nsmap):
        wiki_titles.append(title.text.lower())


    for text in root.findall(".//text", namespaces=root.nsmap):
        wiki_texts.append(text.text.lower())


    wiki_dic = dict(zip(wiki_titles, wiki_texts))
    return wiki_dic, wiki_titles




def regex_for_text_smoothing(wiki_dic, wiki_titles):
    """ A function using regex for smoothing the texts to get rid of brackets 
    and stuff.

    Parameters
    ----------
    wiki_dic : dict
        Contains the extracted titles (key) and texts (value) of Wikipedia articles.
    wiki_titles : list
        Contains list of titles of Wikipedia articles.
    
    Returns
    -------
    wiki_dic : dict
        Contains the smoothed titles (key) and texts (value) of Wikipedia articles without brackets and other things.
    """
    regex = [r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',r'\[\[Datei(.*?)\]\]',r'\<ref(.*?)ref\>', r'z\.B\.', r'\b\w{1,2}\b', r'\[\[(.*?)\|', r'\[', r'\]', r'\<(.*?)\>', r'\{\{(.*?)\}\}', r'(\=\=\sSiehe\sauch\s\=\=)(?s)(.*$)', r'\'\'', r'\=\=\s',  r'\s\=\=', r'\*\s', r'\&nbsp\;', r'\'']

    for r in regex:
        pattern = re.compile(r)
        for title in wiki_titles:
            wiki_dic[title] = re.sub(pattern,'', wiki_dic[title])
    return wiki_dic




def tok_lemmatizing(wiki_dic, title):
    """ A function that tokenizes and lemmatizes the data. 

    Parameters
    ----------
    wiki_dic : dict
        Contains the smoothed titles (key) and texts (value) of Wikipedia articles without brackets and other things.
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
        doc = nlp(word)
        result = ' '.join([x.lemma_ for x in doc]) 
        data_lemma.append(result)

    return data_lemma



def remove_stopwords(data_lemma, manual=True):
    """ A function that removes stop words from data and finding frequencies 
    of words.

    Parameters
    ----------
    data_lemma : list
        Contains the Wikipedia texts in tokenized and lemmatized form.

    Returns
    -------
    text : str
        Contains the Wikipedia text without stopwords. 
    """
    german_stop_words = stopwords.words('german')
 
    if manual == True:
        new_data=[]

        for word in data_lemma:
            if word not in german_stop_words:
                new_data.append(word)
        counts = Counter(new_data)
        newdata = dict(counts)
        main_topic= heapq.nlargest(5, newdata, key=newdata.get)
        return main_topic

    else:
        #for word in data_lemma:
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
        wiki_dic, wiki_titles = extracting_titles_and_texts(name)
        wiki_dic = regex_for_text_smoothing(wiki_dic, wiki_titles)
        for title in wiki_titles:
            titles.append(title)
            texts.append(remove_stopwords(tok_lemmatizing(wiki_dic, title), manual=False))
    
    corpus = dict(zip(titles, texts))

    with open('corpus.json', 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)




filenames = ["WikipediaZeichnen.xml", "WikipediaSportarten.xml", "WikipediaKlettern.xml"]