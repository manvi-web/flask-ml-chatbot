from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
df = pd.read_csv("effort_manual_data.csv")
texts = df["Description"].fillna("").astype(str).tolist()
titles = df["Section"].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question")
    if not user_question:
        return jsonify({"answer": "Please ask a question."})

    user_tfidf = vectorizer.transform([user_question])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    best_match_index = cosine_sim.argmax()
    best_score = cosine_sim[0][best_match_index]

    if best_score > 0.1:
        answer = texts[best_match_index]
        section = titles[best_match_index]
        return jsonify({"answer": f"📖 From *{section}*:\n\n{answer}"})
    else:
        return jsonify({"answer": " Sorry, I couldn't find a relevant answer."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
