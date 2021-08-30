import preprocessing as pp
import json
import click

def calulate_accuracy(frequency_based):
    """ A function that calculates the accuracy of the frequency-based method.

    Parameters
    ----------
    frequency_based : dict
        Contains the Wikipedia titles (keys) and five most frequently used words (values).

    Returns
    -------
    finalcount/247*100 : float
        'finalcount' contains number of cases, in which the title occured in at least one 
        of the five keywords. 247 is the number of all Wikipedia articles.
    """
    number_wikiarticles = 247
    titles = list(frequency_based.keys())
    keywords = list(frequency_based.values())

    finalcount = 0
    for i in range(len(titles)):
        tmp_count = 0
        for key in keywords[i]:
            if key in titles[i]:
                tmp_count += 1
        if tmp_count >= 1:
            finalcount += 1

    return finalcount/number_wikiarticles*100


def frequency_based_extraction():
    """ A function that extracts the five most frequently keywords (lemmas) of 
    the Wikipedia texts and stores them together with the 
    titles in a json file.

    Parameters
    ----------
    -

    Returns
    -------
    None
    """
    keyword_list = []
    title_list = []
    for name in pp.filenames:
        wiki_dic, wiki_titles = pp.extracting_titles_and_texts(name)
        wiki_dic = pp.regex_for_text_smoothing(wiki_dic, wiki_titles)
        for title in wiki_titles:
            title_list.append(title)
            keyword_list.append(pp.remove_stopwords(pp.tok_lemmatizing(wiki_dic, title)))

    final = dict(zip(title_list, keyword_list))     

    with open('frequency-based_extraction.json', 'w', encoding='utf-8') as f:
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
    f = open('frequency-based_extraction.json', encoding='utf-8')
    frequency_based = json.load(f)
    print("\nTotal Accuracy: ", calulate_accuracy(frequency_based))