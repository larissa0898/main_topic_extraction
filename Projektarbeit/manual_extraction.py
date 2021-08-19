import preprocessing_text as pt
import json

final_list = []
title_list = []

for name in pt.filenames:
    wiki_dic, wiki_titles = pt.extracting_titles_and_texts(name)
    wiki_dic = pt.regex_for_text_smoothing(wiki_dic, wiki_titles)
    for title in wiki_titles:
        #print("original topic: ", title, "               predicted topics: ", remove_stopwords(tok_lemmatizing(wiki_dic, title)))
        title_list.append(title)
        final_list.append(pt.remove_stopwords(pt.tok_lemmatizing(wiki_dic, title)))

final = dict(zip(title_list, final_list))     

with open('manual_extraction.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=4)