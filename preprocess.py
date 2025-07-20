from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

def preprocess_medquad():
    # Load MedQuAD dataset
    data = pd.read_csv("medquad.csv")  
    questions = data['question'].tolist()
    answers = data['answer'].tolist()

    # Generate embeddings
    print("Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained sentence transformer
    embeddings = model.encode(questions)

    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save the index
    faiss.write_index(index, "medquad_index.faiss")
    print("Index saved as 'medquad_index.faiss'")

if __name__ == "__main__":
    preprocess_medquad()
