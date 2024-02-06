import chromadb
from chromadb.config import Settings


def split_to_chunks(filepath = None, chunk_size = 8, overlap = 2):
    """  
    split .txt file to chunks
    """
    
    with open(filepath, 'r+', encoding = 'utf8') as f:
        lines = f.readlines()
        f.close()
    n = len(lines)
    chunks = [''.join(lines[i: min(n-1, i+chunk_size)]) for i in range(0,n,chunk_size-overlap)]

    return chunks

text_path = r'C:\Users\spari\OneDrive\Desktop\askTI\output_text_with_ques.txt'

client = chromadb.PersistentClient(path="chromaDB/")

collection = client.create_collection(name="FAQs_v9")
chunks = split_to_chunks(text_path)
n = len(chunks)
collection.add(
    documents = split_to_chunks(text_path),
    ids = list(map(str,range(n)))
    )


