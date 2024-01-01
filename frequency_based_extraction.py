import preprocessing as pp
import json
import click
from collections import Counter
import heapq

def calulate_accuracy(frequency_based, number_wikiarticles):
    """ A function that calculates the accuracy of the frequency-based method.

    Parameters
    ----------
    frequency_based : dict
        Contains the Wikipedia titles (keys) and five most frequently used words (values).
    number_wikiarticles : int
        Contains the number of all Wikipedia articles.

    Returns
    -------
    finalcount/number_wikiarticles*100 : float
        'finalcount' contains number of cases, in which the title occured in at least one 
        of the five keywords.
    """
    titles = list(frequency_based.keys())
    keywords = list(frequency_based.values())

    finalcount = 0
    
    for i in range(len(titles)):
        tmp_count = 0
        for key in keywords[i]:
            if key in titles[i] or titles[i] in key:
                tmp_count += 1
        if tmp_count >= 1:
            finalcount += 1

    return finalcount/number_wikiarticles*100


def frequency_based_extraction():
    """ A function that extracts the five most frequently keywords of 
    the Wikipedia texts and stores them together with the titles in 
    a json file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    keyword_list = []
    title_list = []
    
    for name in pp.filenames:
        wiki_dic = pp.extracting_titles_and_texts(name)
        wiki_dic = pp.regex_for_text_smoothing(wiki_dic)
        for title in wiki_dic:
            title_list.append(title)

            counts = Counter(pp.remove_stopwords(pp.tok_lemmatizing(wiki_dic, title)))
            newdata = dict(counts)
            keyword_list.append(heapq.nlargest(5, newdata, key=newdata.get))

    final = dict(zip(title_list, keyword_list)) 

    with open('frequency_based_extraction.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=4)



###################################################################################################
# Do you want to extract the data with the frequency-based method again and create a new json file? 
# Y - frequency-based extraction starts and new json file is generated
# n - programm runs further with existing json file
####################################################################################################

if click.confirm('Do you want to extract the data with the frequency-based method again and create a new json file?', default=True):
    frequency_based_extraction()

##########################################
# Do you want to calculate the accuracy? 
# Y - calculates accuracy
# n - programm ends
##########################################

if click.confirm('Do you want to calculate the accuracy?', default=True):
    f = open('frequency_based_extraction.json', encoding='utf-8')
    frequency_based = json.load(f)
    print("\nTotal Accuracy: ", calulate_accuracy(frequency_based, number_wikiarticles=202), "%")