from torch.utils.data import DataLoader
from datasets import Dataset
from datasets import load_dataset
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from datasets import concatenate_datasets
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser
import numpy as np
import os
from dotenv import load_dotenv


def search(text_input):   
    #--- variables ---

    load_dotenv()
    storage = os.getenv("invertedIndex")
    ix = open_dir(storage)
    bm25_max = 0
    bm25_avg = 0
    bm25_variance = 0
    top = []


    #--- Get bm25 values + top 3 links from the search --
    try:
        searcher = ix.searcher()

        q = MultifieldParser(["title^2", "content"], ix.schema)
        q = q.parse(text_input)

        results = searcher.search(q, limit= 20)

    
        bm25scores = []
        
        ranklimit = 3 # only top 3

        for i, r in enumerate(results):
  
            bm25scores.append(r.score)
            if(i < ranklimit - 1):
                top.append(r["link"])

 
        bm25 = np.array(bm25scores)

        bm25_max = np.max(bm25)
        bm25_avg = np.mean(bm25)
        bm25_variance = np.var(bm25)

        print(bm25_max, bm25_avg, bm25_variance)
    finally:
        searcher.close()
        return bm25_max, bm25_avg, bm25_variance, top
            




def Data(BATCH_SIZE):

    #--- load in data set ---
    MODEL_NAME = "roberta-base"

    #load in
    real = os.getenv("real")
    fake = os.getenv("fake")
    dataset = load_dataset("text", data_files={ "real": rf"{real}",
                                                "fake": rf"{fake}" } )

    #label
    real_dataset = dataset["real"].add_column("labels", [1] * len(dataset["real"]))
    fake_dataset = dataset["fake"].add_column("labels", [0] * len(dataset["fake"]))
    
    df = concatenate_datasets([real_dataset, fake_dataset])


    # Split into train / validation
    indices = list(range(len(df)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.1,
        random_state=42,
        stratify=df["labels"]
    )

    #turn into objects
    train_dataset = df.select(train_indices)
    val_dataset = df.select(val_indices)

    #take only a subset of the data
    train_subset = train_dataset.shuffle(seed=42).select(range(250))
    val_subset = val_dataset.shuffle(seed=42).select(range(250))       



    #--- add bm25 scores to the data ---
    def add_bm25(x):
        query = x["text"][:500]
        bm25_max, bm25_avg, bm25_variance, _ = search(query)
        x["extra_features"] = [bm25_max, bm25_avg, bm25_variance]    

        return x

    train_subset = train_subset.map(add_bm25)
    val_subset = val_subset.map(add_bm25)


    #--- tokenize all the data ---

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        MAX_LENGTH = 384
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    train_subset = train_subset.map(tokenize_function, batched=True)
    val_subset = val_subset.map(tokenize_function, batched=True) 

    #tokenization does not replace columns, so we remove previous ones
    train_subset = train_subset.remove_columns(["text"])
    val_subset = val_subset.remove_columns(["text"])


    # Set format for PyTorch
    train_subset.set_format("torch")
    val_subset.set_format("torch")

    
    #--- create and return data loader objects ---
    train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE
    )


    return train_loader, val_loader



