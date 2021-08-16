from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pandas
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
import json
import click
from preprocessing_text import save_corpus_in_json


def get_top_n_words(corpus, n=None):
    """ A function that returns the most frequently occuring words.. 

    Parameters
    ----------
    corpus : list
        Contains the texts of the Wikipedia articles.
    n : None
        Contains a title of the list of Wikipedia topics.

    Returns
    -------
    words_freq : list[tuple]
        Contains the most frequent words.
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]


def sort_coo(coo_matrix):
    """ A function that sorts tf_idf in descending order. 

    Parameters
    ----------
    coo_matrix : tuple/matrix
        

    Returns
    -------
    tuples : tuple
        Returns sorted matrix.
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """ A function that gets the feature names and tf-idf score of top n items.
    
    Parameters
    ----------
    feature_names : list
        Contains feature names.
    sorted_items : list[Tuple]
        Contains 
    topn : int
        Contains a number, that defines the top n items of the vector that should be used.
    
    Returns
    -------
    results : dict
        Contains 
    """
    
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
        results[feature_vals[idx]] = score_vals[idx]
    
    return results

#############################################################################
# Do you want to store the data in a json file or load the current json file? 
# Y - preprocessing start and new json file is generated
# n - programm runs further with current json file
############################################################################# 

if click.confirm('Do you want to store the data in a new json file or load the current json file?', default=True):
    save_corpus_in_json()


f = open('corpus.json', encoding='utf-8')
corpus = json.load(f)



german_stop_words = stopwords.words('german')
cv = CountVectorizer(max_df=0.8, stop_words=german_stop_words, max_features=10000, ngram_range=(1,3))
doc_term_matrix = cv.fit_transform(corpus)



""" # Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=20)
top_df = pandas.DataFrame(top_words)
top_df.columns = ["Word", "Freq"]

# Barplot of most freq words
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30) """

 
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(doc_term_matrix)
# get feature names
feature_names = cv.get_feature_names()
 

for i in range(len(corpus)):
    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([corpus[i]]))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items,5)
 
    # now print the results
    print("\nAbstract:")
    print(corpus[i])
    print("\nKeywords:")
    for k in keywords:
        print(k, keywords[k])


""" 
# fetch document for which keywords needs to be extracted
doc = corpus[0]
 
# generate tf-idf for the given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

# sort the tf-idf vectors by descending order of scores
sorted_items = sort_coo(tf_idf_vector.tocoo())

# extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items,5)
 
# now print the results
print("\nAbstract:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k,keywords[k]) """