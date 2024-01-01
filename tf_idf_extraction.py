from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import json
import click
import sys
from preprocessing import save_corpus_in_json


def sort_coo_matrix(coo_matrix):
    """ A function that sorts the tf-idf vectors by descending order of scores.

    Parameters
    ----------
    coo_matrix : scipy.sparse.coo.coo_matrix
        Contains matrix with encoded words and the scores.
        

    Returns
    -------
    sorted(tuples) : list[Tuple]
        Return sorted list of tuples, which contain the encoded words (=index) and the corresponding scores.
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 

def extract_first_n_from_vector(feature_names, sorted_items, n=5):
    """ A function that extracts the feature names and tf-idf score of top n items.
    
    Parameters
    ----------
    feature_names : list
        Contains feature names.
    sorted_items : list[Tuple]
        Contains the encoded words and scores per article. 
        Has following structure: [(Index of word1, highest_score), (Index of a word2, second_highest_score), ...]
    n : int
        Contains a number, that defines the first n items of the vector that should be used.
    
    Returns
    -------
    results : dict
        Contains the first n keywords (key) with their corresponding scores (values).
    """
    sorted_tuples = sorted_items[:n]


    score_values = []
    feature_values = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_tuples:
        score_values.append(round(score, 3))         # keep track of feature name and its corresponding score
        feature_values.append(feature_names[idx])

    results= {}
    for idx in range(len(feature_values)):
        results[feature_values[idx]] = score_values[idx]

    return results



######################################################################################
# Do you want to store the data in a json file (y) or load the existing json file (n)? 
# Y - preprocessing starts and new json file is generated
# n - programm runs further with existing json file
###################################################################################### 

if click.confirm('Do you want to store the data in a new json file (y) or load the current json file (n)?', default=True):
    save_corpus_in_json()


try:
    with open('tf_idf_extraction.json', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("FileNotFoundError: Sorry, there is no current file yet.")
    sys.exit()


title = list(data.keys())
corpus = list(data.values())



cv = CountVectorizer(max_df=0.8, max_features=10000)
doc_term_matrix = cv.fit_transform(corpus)
 
tfidf_transformer = TfidfTransformer(smooth_idf=True, 
                                    use_idf=True)
tfidf_transformer.fit(doc_term_matrix)

feature_names = cv.get_feature_names_out()



all_articles_keywords = []

for i in range(len(corpus)):
    one_article_keywords = []
    tf_idf_vector = tfidf_transformer.transform(cv.transform([corpus[i]])) 

    sorted_items = sort_coo_matrix(tf_idf_vector.tocoo()) 

    keywords = extract_first_n_from_vector(feature_names, sorted_items, n=5)
        
    print("\nTitel: ", title[i])
    print("Keywords:")
    for k in keywords:
        print(k, keywords[k])

    for k in keywords:
        one_article_keywords.append(k)
    all_articles_keywords.append(one_article_keywords)

realcount = 0
number_wikiarticles = 202

for i in range(len(title)):
    count = 0
    for key in all_articles_keywords[i]:
        if key in title[i] or title[i] in key:
            count += 1
    if count >= 1:
        realcount += 1

print("\nTotal Accuracy: ", realcount/number_wikiarticles*100, "%")