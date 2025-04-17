from flask import Flask, request, jsonify
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("effort_manual_data.csv")
sections = df["section"].tolist()
texts = df["content"].tolist()
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)
app = Flask(__name__)
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    question_vec = vectorizer.transform([question])
    similarity = cosine_similarity(question_vec, tfidf_matrix)
    index = similarity.argmax()

    return jsonify({
        "section": sections[index],
        "answer": texts[index]
    })
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  
    app.run(host='0.0.0.0', port=port, debug=True)