from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
import json
import click
from preprocessing_text import save_corpus_in_json


def sort_coo(coo_matrix):
    """ A function that sorts the tf-idf vectors by descending order of scores.

    Parameters
    ----------
    coo_matrix : tuple/matrix
        

    Returns
    -------
    sorted(tuples) : list[tuple]
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
        Contains the output of function 'sort_coo()'.
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
data = json.load(f)

title = list(data.keys())
corpus = list(data.values())


german_stop_words = stopwords.words('german')
cv = CountVectorizer(max_df=0.8, stop_words=german_stop_words, max_features=10000, ngram_range=(1,3))
doc_term_matrix = cv.fit_transform(corpus)



 
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(doc_term_matrix)
# get feature names
feature_names = cv.get_feature_names()
 
list_list = []

for i in range(len(corpus)):
    tmp_list = []
    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([corpus[i]]))

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items,5)
 
    # now print the results
    print("\nTitel: ", title[i])
    print("Keywords:")
    for k in keywords:
       print(k, keywords[k])
"""     for k in keywords:
        tmp_list.append(k)
    list_list.append(tmp_list)



realcount = 0

for i in range(len(title)):
    count = 0
    for key in list_list[i]:
        if key in title[i]:
            count += 1
    if count >= 1:
        realcount += 1

print(realcount/247*100) """