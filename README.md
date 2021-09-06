# Advanced NLP Project - Main Topic Extraction
This repository contains two methods for main topic extraction with Wikipedia articles. One with tf-idf and the other one is frequency-based (count most frequently words/lemmas in text).  
In the further course, the two methods are compared with each other on the basis of some statistics.

#### Table of Contents
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)


## Data
The data (Wikipedia articles) were generated by [this page](https://de.wikipedia.org/wiki/Spezial:Exportieren). 
Through the page you get the articles sorted by a category, e.g. 'Klettern', written into an XML file.  
The three categories were randomly chosen by me (Klettern, Sportarten, Zeichnen). The articles regarding these categories were chosen by the page, which created the XML-files.  
Some example articles of the total 202 Wikipedia articles, which are used in this project, can be found here:  
- [Freiklettern](https://de.wikipedia.org/wiki/Freiklettern)  
- [Concept Art](https://de.wikipedia.org/wiki/Concept_Art)
- [Wurfsportart](https://de.wikipedia.org/wiki/Wurfsportart)


## Installation 
Download all the files on this GitHub repository.  
With 'pip install -r requirements.txt' all required libraries are installed.  
For this project I used Python Version 3.9.5. The versions of the libraries, which are in requirements.txt, can also be found in just that. 

![Step1](/Projektarbeit/images/step1.PNG)


## Usage
### tf-idf method
If you want to use the tf_idf-method, you have to run the code in 'tf_idf_extraction.py'. While the programm is running, you will be asked, whether you want to use the existing json file with the pre-processed data ('corpus.json') or whether you want to pre-process the data again and create a new json file (this takes about 8 minutes).  
![Step2](/Projektarbeit/images/step2.PNG)  
In the end, the title of the Wikipedia articles will be printed together with the five keywords, that were generated by the tf-idf method. In the last line the accuracy will be printed. 
![Step3](/Projektarbeit/images/step3.PNG)  
  
  
### frequency-based method
If you want to use the frequency-based method, you have to run the code in 'frequency-based_extraction.py'. You will be asked, if you want to extract the data with the frequency-based method again or if you want to use the existing data in the json file 'frequency-based_extraction.json'.  
If you press 'y' a new json file 'frequency-based_extraction.json' will be created with the titles of the Wikipedia articles and the five most frequently occuring words/lemmas in the Wikipedia texts.   
After that you will be asked, if you want to calculate the accuracy for this method. If you press 'y' the total accuracy will be printed.
![Step4](/Projektarbeit/images/step4.PNG)


### Accuracy
Accuracy is calculated by checking whether one or more of the keywords is a substring of the Wikipedia article title, or whether the title is a substring of one of the keywords.  
For example, this ensures that although a title is called "Seilkommando", the keywords are more likely to be include the plural "Seilkommandos", this prediction is accepted as correct.  
Sometimes the title consists of composites and the keywords of the single words that can be combined to form the composite. This method of calculating the accuracy ensures that these cases are also recognized as correct.  
For each article that matches either case, the integer variable count is incremented by one. At the end, the number of Wikipedia articles is calculated by count times 100 to get the accuracy in percent (202/count*100).


## References
[Medium: "Automated Keyword Extraction from Articles using NLP](https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34)  
[PyPi: wiki-dump-reader 0.0.4](https://pypi.org/project/wiki-dump-reader/)