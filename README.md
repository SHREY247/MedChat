# DigiDoctor: AI-Powered Medical Assistant

DigiDoctor is a medical chatbot powered by **Retrieval-Augmented Generation (RAG)**. It uses the **MedQuAD dataset** for retrieval and **Llama3.2** (via Ollama) for generating real-time AI responses.

## Features
- Intelligent medical assistant using a RAG pipeline.
- Provides reliable responses based on the MedQuAD dataset.
- Real-time streaming of AI-generated answers via Ollama.

---

## Prerequisites
- Python 3.8 or later.
- **[Ollama](https://ollama.ai)** installed and running on `http://localhost:11434`.

---

## Installation and Setup

### 1. Clone this repository:
git clone https://git.digimantra.com/SHREY/DigiDoctor.git <br />
cd DigiDoctor


## 2. Install dependencies in virtual environment:
python3 -m venv venv  <br />
source venv/bin/activate   <br />
pip install -r requirements.txt


## 3. Start the Ollama server:
curl -fsSL https://ollama.com/install.sh | sh <br />
ollama serve


## 4. (Optional) Generate FAISS index if not already present:
If the medquad_index.faiss file is missing or needs to be rebuilt: <br />
python3 preprocess.py


## 5. Running the Application
Start the Streamlit app: <br />
streamlit run app.py