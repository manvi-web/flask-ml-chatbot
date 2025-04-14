from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
app = Flask(__name__)
CORS(app)
df = pd.read_csv("faq_dataset.csv")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question'])
@app.route('/')
def index():
    return render_template('index.html') 
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"answer": "Please provide a valid question."})
    
    input_vec = vectorizer.transform([question])
    similarity = cosine_similarity(input_vec, tfidf_matrix)
    idx = similarity.argmax()
    answer = df.iloc[idx]["answer"]
    
    return jsonify({"answer": answer})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render gives the port via $PORT
    app.run(host='0.0.0.0', port=port)
