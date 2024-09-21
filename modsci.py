import pandas as pd
import networkx as nx
import json
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import gensim.downloader as api
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from rake_nltk import Rake

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load pre-trained word embeddings
word_vectors = api.load('fasttext-wiki-news-subwords-300')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]

def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:10]  # Get top 10 keywords/phrases

def get_word_vector(word):
    if word in word_vectors:
        return word_vectors[word]
    return np.zeros(300)  # Return zero vector if word not in vocabulary

def create_keyword_vectors(keywords):
    return np.array([get_word_vector(keyword) for keyword in keywords])

def cluster_keywords(keyword_vectors):
    dbscan = DBSCAN(eps=0.4, min_samples=6)
    clusters = dbscan.fit_predict(keyword_vectors)
    return clusters

def get_word_vector(phrase):
    words = phrase.split()
    if len(words) == 1:
        return word_vectors[phrase] if phrase in word_vectors else np.zeros(300)
    else:
        return np.mean([word_vectors[word] for word in words if word in word_vectors] or [np.zeros(300)], axis=0)

def create_topic_model(keywords, n_topics=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(keywords)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    
    topic_keywords = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topic_keywords.append(top_keywords)
    
    return topic_keywords

def create_mindmap(keywords):
    preprocessed_keywords = [' '.join(preprocess_text(kw)) for kw in keywords]
    extracted_keywords = [item for sublist in [extract_keywords(kw) for kw in preprocessed_keywords] for item in sublist]
    
    keyword_vectors = create_keyword_vectors(extracted_keywords)
    clusters = cluster_keywords(keyword_vectors)
    topics = create_topic_model(extracted_keywords)
    
    G = nx.Graph()
    
    for i, keyword in enumerate(extracted_keywords):
        G.add_node(keyword, cluster=int(clusters[i]))
    
    # Add edges based on similarity
    for i in range(len(extracted_keywords)):
        for j in range(i + 1, len(extracted_keywords)):
            similarity = np.dot(keyword_vectors[i], keyword_vectors[j]) / (np.linalg.norm(keyword_vectors[i]) * np.linalg.norm(keyword_vectors[j]))
            if similarity > 0.9:  # Adjust threshold as needed
                G.add_edge(extracted_keywords[i], extracted_keywords[j], weight=float(similarity))  # Convert to float
    
    # Convert to hierarchical format
    nodes = [{"id": node, "cluster": data['cluster']} for node, data in G.nodes(data=True)]
    edges = [{"source": u, "target": v, "weight": float(data['weight'])} for u, v, data in G.edges(data=True)]
    
    return {
        "nodes": nodes,
        "edges": edges,
        "topics": topics,
        "root": extracted_keywords[0] if extracted_keywords else None
    }

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