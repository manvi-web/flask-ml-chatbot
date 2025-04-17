from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
df = pd.read_csv("effort_manual_data.csv")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["content"].fillna(""))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get-answer', methods=['POST'])
def get_answer():
    user_question = request.form['question']
    question_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(question_vec, X)
    most_similar_idx = similarities.argmax()
    answer = df.iloc[most_similar_idx]["content"]
    return render_template('index.html', answer=answer, question=user_question)
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
