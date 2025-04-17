import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)
df = pd.read_csv("effort_manual_data.csv")  
df.columns = ["Section", "Link", "Content"]
texts = df["Content"].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get_answer", methods=["POST"])
def get_answer():
    user_input = request.form["question"]
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    best_idx = similarity.argmax()
    best_score = similarity[0, best_idx]

    if best_score < 0.1:
        return jsonify({"answer": "Sorry, I couldn't find a relevant answer."})
    
   
    answer = {
        "section": df.iloc[best_idx]["Section"],
        "link": df.iloc[best_idx]["Link"],
        "content": df.iloc[best_idx]["Content"]
    }

    return jsonify(answer)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
