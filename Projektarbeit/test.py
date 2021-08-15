from sklearn.feature_extraction.text import CountVectorizer
import re
import seaborn as sns
import pandas
from lxml import etree
from gensim.utils import tokenize
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer



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

    # iterate through all the titles
    for title in root.findall(".//title", namespaces=root.nsmap):
        wiki_titles.append(title.text)

    # iterate through all the texts
    for text in root.findall(".//text", namespaces=root.nsmap):
        wiki_texts.append(text.text)


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
    regex = [r'\[\[Datei(.*?)\]\]',r'\<ref(.*?)ref\>', r'z\.B\.', r'\b\w{1,2}\b', r'\[\[(.*?)\|', r'\[', r'\]', r'\<(.*?)\>', r'\{\{(.*?)\}\}', r'(\=\=\sSiehe\sauch\s\=\=)(?s)(.*$)', r'\'\'', r'\=\=\s',  r'\s\=\=', r'\*\s', r'\&nbsp\;', r'\'']

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






def remove_stopwords(data_lemma):
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

    #for word in data_lemma:
    text = [word for word in data_lemma if not word in  
            german_stop_words] 
    text = " ".join(text)
    return text.lower()


filenames = ["WikipediaZeichnen.xml"]   #, "WikipediaSportarten.xml", "WikipediaKlettern.xml"
corpus=[]

for name in filenames:
    wiki_dic, wiki_titles = extracting_titles_and_texts(name)
    wiki_dic = regex_for_text_smoothing(wiki_dic, wiki_titles)
    for title in wiki_titles:
        corpus.append(remove_stopwords(tok_lemmatizing(wiki_dic, title)))

#print(corpus[0])

german_stop_words = stopwords.words('german')
cv=CountVectorizer(max_df=0.8,stop_words= german_stop_words, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)



#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=20)
top_df = pandas.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

#Barplot of most freq words

""" sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30) """

 
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X)
# get feature names
feature_names=cv.get_feature_names()
 
# fetch document for which keywords needs to be extracted
doc=corpus[2]
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))


#Function for sorting tf_idf in descending order
from scipy.sparse import coo_matrix
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """ get the feature names and tf-idf score of top n items """
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,5)
 
# now print the results
print("\nAbstract:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k,keywords[k])