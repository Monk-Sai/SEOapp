import pandas as pd
import io
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for Matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
from nltk.corpus import wordnet
import nltk

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

    # Create the graph image
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=8, node_size=3000, font_weight='bold')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)

        # Debug: Check if 'Keywords' column exists
        if 'Keywords' not in df.columns:
            return "Error: 'Keywords' column not found in the uploaded CSV file."

        keywords = df['Keywords'].tolist()

        # Generate the mind map image
        img = create_mindmap(keywords)

        return send_file(img, mimetype='image/png', as_attachment=False, download_name='mindmap.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
