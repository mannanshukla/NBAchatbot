# Import necessary libraries
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

# Assuming Ollama is running locally and available at a specific URL, e.g., http://localhost:5000
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0, base_url="http://127.0.0.1:11434")
# Use HuggingFaceEmbeddings with LangchainEmbedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embed_model = LangchainEmbedding(embedding_model)
Settings.embed_model = embed_model

# Load CSV data from Basketball Reference using pandas
csv_path = "ref.csv"  # Replace with your actual CSV file path
data = pd.read_csv(csv_path)
print(data)  # Print the DataFrame to verify it loaded correctly

# Convert each row of the DataFrame into a Document for Llama Index
documents = [Document(text=row.to_string()) for _, row in data.iterrows()]

# Create the VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# Initialize a query engine from the index
query_engine = index.as_query_engine()

# Function to interact with the query engine and use Llama 3.2 for responses
def chat_with_index(query):
    # Use the query engine to get relevant context
    response = query_engine.query(query)
    return response.response

# Example queries you can use
print(chat_with_index("Who was the absolute worst player this season?"))
