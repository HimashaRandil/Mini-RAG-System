import os
import json
import openai
from dotenv import load_dotenv
from utils.logger.logging import logger as logger

# Import our custom utility functions
from src.pipeline import (
    load_and_preprocess_data,
    chunk_documents,
    create_vector_store,
)

# Load environment variables from .env file in the root directory
load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")
client = openai.OpenAI(api_key=api_key)


def query_rag_system(query, collection, embedding_model):
    """
    Retrieves relevant context and generates a structured answer using an LLM.
    """
    logger.info(f"\nProcessing query: '{query}'")
    query_embedding = list(embedding_model.embed([query]))[0]

    retrieved_results = collection.query(
        query_embeddings=[query_embedding], n_results=3
    )
    retrieved_contexts = retrieved_results["documents"][0]

    prompt = f"""
    You are a helpful movie plot assistant. Based on the following context, please answer the user's question.

    Your response MUST be a single JSON object with the following three keys:
    - "answer": A natural language answer to the question.
    - "contexts": A list of the retrieved plot snippets you used.
    - "reasoning": A short explanation of how you formed the answer from the context.

    Your response MUST be a single JSON object. Follow the structure of this example:
    {{
    "answer": "A natural language answer to the question.",
    "contexts": [
        "The first retrieved plot snippet...",
        "The second retrieved plot snippet..."
    ],
    "reasoning": "A short explanation of how you formed the answer from the context."
    }}


    CONTEXT:
    ---
    {"---".join(retrieved_contexts)}
    ---

    QUESTION:
    {query}
    """

    logger.info("Generating answer with OpenAI GPT-4o...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    final_output = json.loads(response.choices[0].message.content)
    return final_output


if __name__ == "__main__":
    # The path is now relative to the root directory
    DATA_PATH = "data/wiki_movie_plots_deduped.csv"

    documents = load_and_preprocess_data(DATA_PATH)
    chunks = chunk_documents(documents)
    collection, embedding_model = create_vector_store(chunks)

    user_query = "Which movie is about a beautiful girl who is forced into hiding?"
    rag_output = query_rag_system(user_query, collection, embedding_model)

    logger.info("\n--- RAG System Output ---")
    logger.info(json.dumps(rag_output, indent=2))
    logger.info("--- End of Output ---")
