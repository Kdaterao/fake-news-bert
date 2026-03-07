import pandas as pd
import os
from whoosh import index
from whoosh.index import create_in, open_dir
from whoosh.fields import *



def main():

    #-----Variables-----
    csv_files = ["/Users/krish/codingStuff/credibility-tracker/results/Google/newspaper.csv","/Users/krish/codingStuff/credibility-tracker/results/Twitter/newspaper.csv"]
    schema = Schema(id=ID(unique=True), link=ID(stored=True), title=TEXT(stored=True), content=TEXT(stored=True)) 
    inverted_storage = "invertedIndex1517"
    last_chunk_file = "lastChunk.txt"
    currentChunk = 0
    MAX_CONTENT_LENGTH = 5000
    frequency = 4 #we only want a fraction of the data, so we use this frequency variable 





    #-----Get last chunk processed(in event that the code breaks)-----
    
    if os.path.exists(last_chunk_file):
        with open(last_chunk_file, "r") as f:
            last_chunk = int(f.read().strip())
    else:
        last_chunk = 0  # start from beginning
    




    #-----Get or create our inverted index object------
    
    if not os.path.exists(inverted_storage):
        os.mkdir(inverted_storage)
        ix = index.create_in(inverted_storage, schema)
    
    else:
        ix = open_dir(inverted_storage)
    


    #-----get length -----

    length = 0
    for file in csv_files:

        #---iterate over it by chunk---
        for chunk in pd.read_csv(file, chunksize=5000):
            length += 1

    print(length)


    #-----Build inverted index-----

    writer = ix.writer()


    for file in csv_files:

        #---iterate over it by chunk---
        for chunk in pd.read_csv(file, chunksize=5000):
            currentChunk += 1

    
            #---go to last used chunk---
            if currentChunk < last_chunk: #starts at index we stopped at
                continue

            #---Add documents---
            for i, row in enumerate(chunk.itertuples(index=False), 1): #intertuples faster than interrows

                try:
                    try:
                        if i%frequency != 0:
                            continue
                        
                        if(row.id is None or row.link is None or row.title is None or row.content is None):
                            continue
                        doc_id = row.id
                        doc_link =  row.link
                        doc_title = row.title
                        doc_content = row.content[:MAX_CONTENT_LENGTH]
                        writer.add_document(id= doc_id, link= doc_link, title= doc_title, content= doc_content) #Whoosh automatically replaces documents that already exists(since id is Unique), so no need to account for that

                    except Exception as e:
                        print("Error adding document:", e)
                        continue
                #---Commit current writer object in batches---
                finally:
                    try:
                        #bruh we cant batch more than one because whoosh does not handle large pieces of text well
                        if i%frequency == 0:
                            writer.commit()
                            writer = ix.writer()
                            print("Chunk:", currentChunk," Processed!")

                            with open(last_chunk_file, "w") as f:
                                f.write(str(currentChunk))

                    except Exception as e:
                        writer.cancel()
                        writer = ix.writer()
                        print("Error:", e)


    




if __name__ == "__main__":
    main()
