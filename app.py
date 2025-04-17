from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
df = pd.read_csv("effort_manual_data.csv")
texts = df["content"].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        question_vec = vectorizer.transform([question])
        similarity = cosine_similarity(question_vec, X)
        best_match_index = similarity.argmax()
        answer = texts[best_match_index]
    return render_template("index", answer=answer)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
