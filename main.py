from llama_index.llms.ollama import Ollama
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
from flask import Flask, request, render_template_string, session, redirect, url_for
import re

# Configure Llama 3.2 (Ollama) settings
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0, base_url="http://127.0.0.1:11434")

# Set up the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embed_model = LangchainEmbedding(embedding_model)
Settings.embed_model = embed_model

# Load CSV data from Basketball Reference
csv_path = "new.csv"  # Replace with your actual CSV file path
data = pd.read_csv(csv_path)

# Convert each row of the DataFrame into a Document for Llama Index
documents = [Document(text=row.to_string()) for _, row in data.iterrows()]

# Create the VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)

# Initialize the query engine
query_engine = index.as_query_engine()

# Function to interact with the query engine and use Llama 3.2 for responses
def chat_with_index(query):
    response = query_engine.query(query)
    return response.response

# Helper function to extract player names and create an image URL
def get_player_image_url(response):
    # Match any player names (assuming format "First Last")
    match = re.search(r"\b([A-Z][a-z]+) ([A-Z][a-z]+)\b", response)
    if match:
        first_name, last_name = match.groups()
        initials = f"{last_name[:5].lower()}{first_name[:2].lower()}01"
        return f"https://www.basketball-reference.com/req/202106291/images/players/{initials}.jpg"
    return None

# Initialize Flask app and configure secret key for sessions
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ...
# Enhanced HTML template with NBA logo background pattern and extended chat container
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Chatbot</title>
    <style>
        /* Reset CSS */
        * { margin: 0; padding: 0; box-sizing: border-box; }

        /* Body styling with NBA logo pattern */
        body {
            font-family: Arial, sans-serif;
            background-color: #1e3c72;
            background-image: url('{{ url_for('static', filename='nba_logo.png') }}');
            background-repeat: repeat;
            background-size: 100px 100px; /* Adjust size to make the logo smaller or larger */
            color: #333;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 20px;
        }

        /* Main container styling */
        .container {
            max-width: 600px;
            width: 100%;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        h1 { color: #1e3c72; font-size: 1.8em; margin-bottom: 15px; }

        /* Chat container - extended height */
        .chat-container {
            max-height: 600px;  /* Increased height to show more messages */
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f9f9f9;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .message { margin-bottom: 10px; text-align: left; display: flex; align-items: flex-start; }

        .message.user {
            color: #1e3c72;
            font-weight: bold;
        }

        .message.bot {
            color: #333;
            display: flex;
            align-items: flex-start;
        }

        /* Bot PFP and Player image styling */
        .bot-pfp {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px;
            object-fit: cover;
        }

        .response-text {
            display: flex;
            flex-direction: column;
        }

        .player-image {
            width: 150px;
            height: auto;
            border-radius: 5px;
            margin-top: 5px;
        }

        /* Form styling */
        form {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .button-container {
            display: flex;
            gap: 10px;
        }

        input[type="text"] {
            width: 100%;
            padding: 10px;
            font-size: 1em;
            border: 1px solid #ddd;
            border-radius: 5px;
            transition: border 0.3s;
        }

        input[type="text"]:focus {
            outline: none;
            border-color: #1e3c72;
        }

        input[type="submit"], .clear-btn {
            padding: 10px 20px;
            background-color: #1e3c72;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background-color 0.3s;
        }

        input[type="submit"]:hover, .clear-btn:hover { background-color: #2a5298; }
    </style>
</head>
<body>
    <audio autoplay loop>
        <source src="{{ url_for('static', filename='background.mp3') }}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>

    <div class="container">
        <h1>NBA Chatbot</h1>
        <div class="chat-container" id="chat-container">
            {% for entry in chat_history %}
                <div class="message user">User: {{ entry['query'] }}</div>
                <div class="message bot">
                    <img src="{{ url_for('static', filename='pfp.png') }}" alt="Bot Profile Picture" class="bot-pfp">
                    <div class="response-text">
                        <strong>Floyd.AI:</strong> {{ entry['response'] }}
                        {% if entry['player_image_url'] %}
                            <img src="{{ entry['player_image_url'] }}" alt="Player Image" class="player-image">
                        {% endif %}
                    </div>
                </div>
            {% endfor %}
        </div>
        <form method="POST">
            <div class="button-container">
                <input type="text" id="query" name="query" placeholder="Type your question..." required>
                <input type="submit" value="Send">
            </div>
        </form>
        <form action="/clear_chat" method="POST">
            <button type="submit" class="clear-btn">Clear Chat</button>
        </form>
    </div>
</body>
</html>
"""

# Route for the homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == "POST":
        query = request.form["query"]
        response = chat_with_index(query)
        player_image_url = get_player_image_url(response)
        
        session['chat_history'].append({
            'query': query,
            'response': response,
            'player_image_url': player_image_url
        })
        session.modified = True

    return render_template_string(html_template, chat_history=session['chat_history'])

# Route to clear chat history
@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session.pop('chat_history', None)  # Remove chat history from session
    return redirect(url_for('index'))  # Redirect back to the main page

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
