from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
import json
import click
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
        Return sorted list of tuples, that contain the encoded word (=index) and the corresponding score.
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

if click.confirm('Do you want to store the data in a new json file or load the current json file?', default=True):
    save_corpus_in_json()



f = open('corpus.json', encoding='utf-8')
data = json.load(f)

title = list(data.keys())
corpus = list(data.values())

german_stop_words = stopwords.words('german')
cv = CountVectorizer(max_df=0.8, stop_words=german_stop_words, max_features=10000, ngram_range=(1,3))
doc_term_matrix = cv.fit_transform(corpus)
 
tfidf_transformer = TfidfTransformer(smooth_idf=True, 
                                    use_idf=True)
tfidf_transformer.fit(doc_term_matrix)

feature_names = cv.get_feature_names()



list_all_Articles_keywords = []
for i in range(len(corpus)):
    list_article_keywords = []
    tf_idf_vector = tfidf_transformer.transform(cv.transform([corpus[i]])) 

    sorted_items = sort_coo_matrix(tf_idf_vector.tocoo()) 

    keywords = extract_first_n_from_vector(feature_names, sorted_items, n=5)
        
    #print("\nTitel: ", title[i])
    #print("Keywords:")
    #for k in keywords:
    #    print(k, keywords[k])

    for k in keywords:
        list_article_keywords.append(k)
    list_all_Articles_keywords.append(list_article_keywords)

realcount = 0

for i in range(len(title)):
    count = 0
    for key in list_all_Articles_keywords[i]:
        if key in title[i]:
            count += 1
    if count >= 1:
        realcount += 1

print("\nTotal Accuracy: ", realcount/247*100)