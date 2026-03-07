



# Overview

 * This project serves as a new way to detect fake news

 * It is built on a customized **RoBERTa** model and combines two key signals for classification:
 
 1. **Linguistic patterns** of the text.
 2. **BM25 scores** of news outlets when querying the content.



# Datasets
 
 * labeled dataset of real and fake news from 2015 to 2017 
 [Kaggle Dataset](https://www.kaggle.com/code/chanchal24/fake-news-detection/input)

 * 3DLNews2: A Three-decade Dataset of US Local News Articles from 1995 to 2024
 [GitHub Repository](https://github.com/wm-newslab/3DLNews2)
    


### Set Up inverted Index 

* Extract news articles from **3DLNews2** spanning 2015 to 2017 and save them as a CSV file.

* Process the dataset using `index_builder.py` to create an **inverted index** with **Whoosh**, enabling fast and efficient search queries based on article content

* Put filepath in .env, with the name "invertedIndex"


### Set Up kaggle dataset

* Go to Kaggle Dataset and download the dataset, then put their file paths in the .env with variable names "real" and "fake"






 

