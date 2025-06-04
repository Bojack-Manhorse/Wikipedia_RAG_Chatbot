from dotenv import load_dotenv
from faiss import IndexFlatL2
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from wikipedia import DisambiguationError, page

import numpy as np
import os

def get_wikipedia_page(query:str):
    """
    Return the content of the wikipedia webpage from `query` as a string.
    """
    try:
        page_content = page(query)
        return page_content.content
    except DisambiguationError:
        raise DisambiguationError


def chunk_text(text: str, chunk_size: int=500, chunk_overlap: int=50) -> list[str]:
    """
    Splits a string into shorter chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def embed_chunks(chunks: list[str], embedding_model) -> np.array:
    """
    Converts all the strings in `chunks` into thier vector representaions from `embedding_model`.
    """
    return embedding_model.encode(chunks, convert_to_numpy=True)

def build_faiss_index(embeddings: np.array):
    """
    Buils a FAISS vector database using `embeddings`.
    """
    dim = embeddings.shape[1]
    index = IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search_chunks(query:str, chunks: list[str], index, embedding_model, num_results: int = 3):
    """
    Uses a FAISS vector database `index` to find the `num_results` most similar strings in `chunks` to `query` according to thier vector representations from `embedding_model`.
    """
    query_embeddings = np.array(embedding_model.encode(query))
    query_embeddings_expanded = np.expand_dims(query_embeddings, axis=0)
    D, I = index.search(query_embeddings_expanded, num_results)
    return [chunks[i] for i in I[0]]


def get_chunks_pipeline(query:str, webpage:str, embedding_model) -> list[str]:
    webpage_chunks = chunk_text(webpage)
    webpage_embeddings = embed_chunks(webpage_chunks, embedding_model)
    faiss_index = build_faiss_index(webpage_embeddings)
    closest_chunks = search_chunks(query, webpage_chunks, faiss_index, embedding_model, 5)
    return closest_chunks


def generate_answer(query: str, context_chunks: list[str], client: OpenAI) -> str:
    """
    Creates an LLM query string using `query` and `context chunks` and sends it to the OpenAI API using `client`.
    """
    context = "\n\n".join(context_chunks)
    prompt = f"Answer the following question using the following context: \n\n Context: \n{context}\n\nQuestion:\n{query}"
    completion = client.chat.completions.create(
        model="gpt-4",
        messages= [{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

app = FastAPI()

@app.post("/ask_question")
def ask_question(wikipedia_query: str, question: str):
    try:
        wikipedia_page = page(wikipedia_query)
    except DisambiguationError:
        return {"message": "Disambihuation Error, enter a more specific query."}
    
    load_dotenv()

    api_key = os.getenv("API_KEY")

    client = OpenAI(api_key=api_key)

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    closest_chunks = get_chunks_pipeline(question, wikipedia_page.content, embedding_model)

    response = generate_answer(question, closest_chunks, client)

    return {"message": f"{response}"}


if __name__ == "__main__":
    pass

    
