import preprocessing_text as pt

for name in pt.filenames:
    wiki_dic, wiki_titles = pt.extracting_titles_and_texts(name)
    wiki_dic = pt.regex_for_text_smoothing(wiki_dic, wiki_titles)
    for title in wiki_titles:
        #print("original topic: ", title, "               predicted topics: ", remove_stopwords(tok_lemmatizing(wiki_dic, title)))
        print(pt.remove_stopwords(pt.tok_lemmatizing(wiki_dic, title)))