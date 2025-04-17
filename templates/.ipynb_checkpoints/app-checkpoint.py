from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the extracted Effort manual content
df = pd.read_csv("effort_manual_data.csv")  # Make sure this is deployed with the app
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['content'].fillna(""))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    q_vec = vectorizer.transform([question])
    sim = cosine_similarity(q_vec, X)
    best_idx = sim.argmax()
    answer = df.iloc[best_idx]["content"]
    return jsonify({"answer": answer})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
