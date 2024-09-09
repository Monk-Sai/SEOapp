import pandas as pd
import networkx as nx
import json
from flask import Flask, render_template, request, jsonify
from nltk.corpus import wordnet
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def train_word2vec_model(keywords):
    stop_words = set(stopwords.words('english'))
    tokenized_keywords = [word_tokenize(kw.lower()) for kw in keywords]
    filtered_keywords = [[word for word in words if word.isalnum() and word not in stop_words] for words in tokenized_keywords]
    model = Word2Vec(sentences=filtered_keywords, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_related_keywords(keyword, model, keywords):
    related_keywords = []
    if keyword in model.wv:
        similar_words = model.wv.most_similar(keyword, topn=5)
        related_keywords = [word for word, similarity in similar_words if word in keywords]
    return related_keywords

def create_mindmap(keywords):
    G = nx.DiGraph()
    model = train_word2vec_model(keywords)

    for keyword in keywords:
        G.add_node(keyword)
        related_keywords = get_related_keywords(keyword, model, keywords)
        for related_keyword in related_keywords:
            G.add_edge(keyword, related_keyword)  # Add edge from keyword to related_keyword

    # Convert to hierarchical format
    nodes = []
    edges = []
    root_nodes = set(G.nodes) - set(n for _, n in G.edges)

    for node in G.nodes:
        nodes.append({"id": node})
    
    for source, target in G.edges:
        edges.append({"source": source, "target": target})

    return {"nodes": nodes, "edges": edges, "root": list(root_nodes)[0] if root_nodes else None}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)

        if 'Keywords' not in df.columns:
            return jsonify({"error": "'Keywords' column not found in the uploaded CSV file."}), 400

        keywords = df['Keywords'].tolist()
        mindmap_data = create_mindmap(keywords)

        return jsonify(mindmap_data)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)