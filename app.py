import pandas as pd
import io
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for Matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from nltk.corpus import wordnet
import nltk
import base64

nltk.download('wordnet')

app = Flask(__name__)

def get_related_keywords(keyword):
    synonyms = []
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def create_mindmap(keywords):
    G = nx.Graph()

    for keyword in keywords:
        G.add_node(keyword)  # Add main keyword
        related_keywords = get_related_keywords(keyword)
        for related_keyword in related_keywords:
            G.add_node(related_keyword)
            G.add_edge(keyword, related_keyword)  # Connect related keywords

    # Convert the graph to a data structure that can be easily serialized to JSON
    nodes = [{"id": n} for n in G.nodes()]
    edges = [{"source": u, "target": v} for u, v in G.edges()]

    return {"nodes": nodes, "edges": edges}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)

        if 'Keywords' not in df.columns:
            return jsonify({"error": "'Keywords' column not found in the uploaded CSV file."}), 400

        keywords = df['Keywords'].tolist()

        # Generate the mind map data
        mindmap_data = create_mindmap(keywords)

        # Debug: Print the generated data
        print(mindmap_data)

        # Return the data as JSON
        return jsonify(mindmap_data)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
