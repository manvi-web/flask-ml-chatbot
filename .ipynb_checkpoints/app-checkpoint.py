from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
def load_faqs():
    return pd.read_csv("faq_dataset.csv")
df = load_faqs()
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])
y = df["answer"]
model = LogisticRegression()
model.fit(X, y)
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"answer": "Please enter a valid question."})
    
    X_input = vectorizer.transform([question])
    prediction = model.predict(X_input)[0]
    prediction = prediction.replace("\n", "<br>") 
    return jsonify({"answer": prediction})
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
