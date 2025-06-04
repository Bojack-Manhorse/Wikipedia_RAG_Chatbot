# Wikipedia RAG chatbot

This is a RAG chatbot build using FastAPI, that retrieves and chunks data from wikipedia articles. We use the `wikipedia` library to extract data from wikipedia articles, a sentence-transformer model to generate chunk embeddings, and a FAISS vector database to search the chunk embeddings. We then feed generated prompts into the OpenAI API.

## Setup (with Poetry)

### Clone the Repository

```bash
git clone https://github.com/Bojack-Manhorse/Wikipedia_RAG_Chatbot
cd wikipedia-rag
```

### Install Dependencies

Ensure poetry is installed: https://python-poetry.org/docs/#installation

Then run

```bash
poetry install
```

### Set the API key

Create a `.env` file in the root directory, and past the following within:

```ini
API_KEY="your_openai_api_key_here"
```

### Run the FastAPI server

```bash
poetry run uvicorn wikipedia_rag.main:app --reload
```

The API will be available at:

`http://127.0.0.1:8000`

And the interactive documentation will be at:

`http://127.0.0.1:8000/docs`

## Dependencies:

- `fastapi` - Create API methods to send and receive queries.

- `uvicorn` - Run the server for the API.

- `openai` - Send LLM prompts to OpenAI.

- `faiss-cpu` - Create a vector database and run a similarity search.

- `sentence-transformers` - Create vector embeddings of sentence chunks.

- `langchain` - Create sentence chunks from a wikipedia article.

- `wikipedia` - Extract data from a wikipedia article.








