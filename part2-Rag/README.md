# RAG Book Assistant — Groq + Hugging Face + ChromaDB

This project uses:
- **Groq** for chat completions (`openai/gpt-oss-120b`)
- **Hugging Face Sentence Transformers** for local embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **ChromaDB** as the persistent vector store
- **Streamlit** for the PDF upload and question-answering UI

## 1. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure the Groq API key

Copy `.env.example` to `.env` and replace the placeholder with your key:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Get a key from https://console.groq.com/keys. Do not share or commit your `.env` file.

## 4. Run the Streamlit app

```bash
streamlit run app.py
```

Upload a PDF and click **Create / Rebuild Vector Database**. The embedding model is downloaded and run locally the first time. Then enter a question.

## 5. Optional command-line workflow

The included `create_database.py` builds a Chroma database from `document loaders/deeplearning.pdf`:

```bash
python create_database.py
python main.py
```

## Important: rebuild old Chroma data

The original project database was created with OpenAI embeddings. OpenAI and Hugging Face embeddings are not interchangeable. Rebuild the database using the new code before querying it. The Streamlit button and `python create_database.py` both rebuild the relevant database.
