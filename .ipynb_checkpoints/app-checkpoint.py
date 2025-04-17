from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
df = pd.read_csv("effort_manual_data.csv")
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Content"].fillna(""))
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get_answer", methods=["POST"])
def get_answer():
    question = request.form["question"]
    question_vec = vectorizer.transform([question])  # Vectorize the question
    similarity = cosine_similarity(question_vec, X)  # Compare with content
    best_match_idx = similarity.argmax()  # Get the most similar match
    best_match_score = similarity[0, best_match_idx]

    
    if best_match_score < 0.1:
        return jsonify({"answer": "Sorry, I couldn't find an answer to that question."})

    
    return jsonify({
        "section": df.iloc[best_match_idx]["Section"],
        "link": df.iloc[best_match_idx]["Link"],
        "content": df.iloc[best_match_idx]["Content"]
    })
if __name__ == "__main__":
    app.run(debug=True, port=10000)
