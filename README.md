# Comparison of two methods for Main Topic Extraction
This repository offers two distinct methods for extracting main topics from Wikipedia articles, focusing on keyword extraction. One method utilizes tf-idf, while the other relies on frequency-based analysis (counting the most frequent words/lemmas in text). This project delves into a comprehensive comparison of these methods through various statistical analyses.

#### Table of Contents
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)


## Data
The repository includes Wikipedia articles obtained via [this page](https://de.wikipedia.org/wiki/Spezial:Exportieren). Using this page, articles can be sorted by category, such as 'Klettern,' and exported into an XML file. Three categories—Klettern, Sportarten, and Zeichnen—were randomly selected. The articles within these categories were chosen by the exporting mechanism, generating respective XML files.
Out of a total of 202 Wikipedia articles utilized in this project, sample articles can be accessed here: 
- [Freiklettern](https://de.wikipedia.org/wiki/Freiklettern)  
- [Concept Art](https://de.wikipedia.org/wiki/Concept_Art)
- [Wurfsportart](https://de.wikipedia.org/wiki/Wurfsportart)


## Installation 
To get started, follow these steps:

1. Ensure you have Python installed. This project is compatible with Python 3.9.5 and above.

2. Clone the repository to your local machine and navigate to the project directory:
    ```bash
    git clone https://github.com/larissa0898/main_topic_extraction.git
    cd main_topic_extraction
    ```

3. Create and activate a virtual environment:
    ```bash
    python -m venv myenv
    myenv\Scripts\activate # On macOS and Linux: source myenv/bin/activate
    ```

4. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    python -m spacy download de_core_news_md
    ```

## Usage
### tf-idf method
If you want to use the tf_idf-method, run the following command:
    ```bash
    python tf_idf_extraction.py
    ```
While the programm is running, you will be asked, whether you want to use the existing json file with the pre-processed data (`tf_idf_extraction.json`) or whether you want to pre-process the data again and create a new json file (this takes about 8 minutes). 
The output will look like the following:
    ```bash
    Titel: title of Wikipedia article
    Keywords:
    keyword 1 tf-idf-value
    keyword 2 tf-idf-value
    keyword 3 tf-idf-value
    keyword 4 tf-idf-value
    keyword 5 tf-idf-value
    ```
In the last line the accuracy will be printed.  
  
  
### frequency-based method
If you want to use the frequency-based method, run the following command:
    ```bash
    python frequency_based_extraction.py
    ```
You will be asked, if you want to extract the data with the frequency-based method again or if you want to use the existing data in the json file `frequency_based_extraction.json`.  
If you press `y` a new json file `frequency_based_extraction.json` will be created with the titles of the Wikipedia articles and the five most frequently occuring words/lemmas in the Wikipedia texts in descending order.   
After that you will be asked, if you want to calculate the accuracy for this method. If you press `y` the total accuracy will be printed.


## References
[Medium: "Automated Keyword Extraction from Articles using NLP"](https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34)  
[PyPi: wiki-dump-reader 0.0.4](https://pypi.org/project/wiki-dump-reader/)