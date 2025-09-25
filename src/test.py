# main.py
import os
import json
import pandas as pd
import chromadb
import openai
from dotenv import load_dotenv
from fastembed import TextEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger.logging import logger as logger

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
# It's good practice to handle the case where the key is missing
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in .env file")
    raise ValueError("OPENAI_API_KEY not found in .env file")
client = openai.OpenAI(api_key=api_key)

logger.info("Setup complete. Libraries imported and API key loaded.")
