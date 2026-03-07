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

* Extract news articles from **3DLNews2** spanning **2015 to 2017** and save them as a CSV file.

* Process the dataset using `index_builder.py` to create an **inverted index** with **Whoosh**, enabling fast and efficient search queries based on article content.

* Add the file path to the `.env` file using the variable name **`invertedIndex`**.


### Set Up kaggle dataset

* Go to the Kaggle dataset and download it.

* Then add the file paths to the `.env` file with the variable names **`real`** and **`fake`**.



# Training 

* For training, this project uses a **GTX 1070 Ti** with **CUDA 12.6**.
* For other setups, it may be necessary to modify the pip installs related to GPU and CUDA versions.






# Production Use

### Setting Up the Chrome Extension

* inside the `fake-news-extension` folder run:

```
npm run build
``` 

* Then use the generated **dist** folder as the Chrome extension code.


### Running the Backend Server

* This project uses **FastAPI**, so run the following command to start `server.py`:

```
uvicorn server:app --reload
```








