from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
df = pd.read_csv("faq_dataset.csv")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question'])
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    input_vec = vectorizer.transform([question])
    similarity = cosine_similarity(input_vec, tfidf_matrix)
    idx = similarity.argmax()
    answer = df.iloc[idx]["answer"]
    return jsonify({"answer": answer})

if __name__ == '__main__':
   
    app.run(port=10000, debug=True)
