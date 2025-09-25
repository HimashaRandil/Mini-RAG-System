# Mini RAG System for Movie Plots 🍿

This project is a lightweight Retrieval-Augmented Generation (RAG) system built to answer questions about movie plots. It uses a subset of the Wikipedia Movie Plots dataset and demonstrates a clean, end-to-end pipeline: **Load → Chunk → Embed → Retrieve → Generate**.

This system was developed as a solution to the "Take-Home Assignment: Applied AI Engineer".

## ✨ Features

- **Efficient Embeddings**: Utilizes **FastEmbed** with ONNX optimization for fast and lightweight text embeddings.
- **In-Memory Vector Store**: Uses **ChromaDB** for a simple and efficient in-memory vector database.
- **High-Quality Generation**: Leverages **OpenAI's GPT-4o** for reliable and context-aware answer generation.
- **Structured Output**: The final output is always a clean JSON object containing the answer, the context used, and the reasoning.
- **Dual-Mode Operation**: Can be run as a standard command-line script or as an interactive Jupyter Notebook with widgets.

---

## 📂 Project Structure

```
MINI-RAG-SYSTEM/
│
├── .venv/                  # Virtual environment
├── data/                   # Dataset storage
├── notebooks/
│   └── demo.ipynb          # Interactive Jupyter Notebook demo
├── src/
│   ├── main.py             # Main script for CLI execution
│   └── pipeline.py         # Data processing and RAG pipeline functions
│
├── .env                    # Environment variables (API key)
├── .env.example            # Example environment file
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

First, clone the project to your local machine:

```bash
git clone https://github.com/HimashaRandil/Mini-RAG-System.git
cd Mini-RAG-System
```

### 2. Set Up the Environment with uv

This project uses uv for fast and efficient package management.

#### A. Install uv (if you haven't already):

```bash
# On macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

#### B. Create the Virtual Environment and Install Dependencies:

```bash
# Create the virtual environment
uv venv

# Activate the environment
# On Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# On macOS/Linux: source .venv/bin/activate

# Install all packages from pyproject.toml
uv sync
```

### 3. Configure Environment Variables

You'll need an OpenAI API key to run the language model.

Make a copy of the example environment file:

```bash
# On Windows
copy .env.example .env

# On macOS/Linux
cp .env.example .env
```

Open the newly created `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY="sk-YourSecretApiKeyHere"
```

### 4. Download the Dataset

1. Download the Wikipedia Movie Plots dataset from [Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots).
2. Unzip the file and place `wiki_movie_plots_deduped.csv` inside the `data/` directory.

---

## ⚙️ Usage

You can run the RAG system in two ways.

### Option 1: Command-Line Interface

This method runs the entire pipeline and prints a single JSON output to the console. The query is hardcoded in the `src/main.py` script.

```bash
python -m src.main
```

### Option 2: Interactive Jupyter Notebook (Recommended)

This is the best way to interact with the system and ask multiple questions.

1. Start the Jupyter Notebook server from your project's root directory:

```bash
jupyter notebook
```

2. In the browser window that opens, navigate to the `notebooks/` directory and open `demo.ipynb`.

3. Run the cells in order from top to bottom.

4. Use the interactive text box at the end of the notebook to enter your questions and get answers.

---
