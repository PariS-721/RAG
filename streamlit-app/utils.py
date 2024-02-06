import chromadb

def change_context():
    return None

def split_to_chunks(filepath = None, chunk_size = 8, overlap = 2):
    """  
    split .txt file to chunks
    """
    
    with open(filepath, 'r+') as f:
        lines = f.readlines()
    n = len(lines)

    chunks = [lines[i: min(n-1, i+chunk_size)] for i in range(0,n,chunk_size-overlap)]

    return chunks

def get_top_k(query):
    """
    fetches top k contexts from a db
    """
    client = chromadb.PersistentClient(path="chromaDB/")
    collection = client.get_collection(name='FAQs_v9')

    res = collection.query(query_texts = query, n_results = 5)

    return res['documents'][0], res['distances'][0]

def augment_prompt(query_str, context_msg):
    
    #multiple customizeable prompts to be built.

    prompt = f""" 
    The original query is as follows: {query_str}
    Given the following context information and not prior knowledge, answer the query.
    ------------
    {context_msg}
    ------------
    Given the context information and not prior knowledge, answer the query.
    Answer: 
    """
    return prompt