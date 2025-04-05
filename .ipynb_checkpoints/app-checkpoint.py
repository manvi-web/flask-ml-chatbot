from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
data = {
    "question": [
        "What is Effort@Spoors?",
        "How can I reset my password?",
        "How do I track my tasks?",
        "Can I use Effort on mobile?",
        "How do I contact support?",
        "How to export my reports?"
    ],
    "answer": [
        "Effort@Spoors is a no-code platform for business process automation.",
        "You can reset your password by clicking 'Forgot Password' on the login screen.",
        "Track your tasks by navigating to the Tasks tab in the dashboard.",
        "Yes, Effort is available on Android and iOS.",
        "You can contact support via support@spoors.in.",
        "Go to Reports section and click on the export button to download."
    ]
}
df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])
y = df["answer"]
model = LogisticRegression()
model.fit(X, y)
app = Flask(__name__)
@app.route("/")
def index():
    return "FAQ Chatbot API is running!"
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    X_input = vectorizer.transform([user_input])
    prediction = model.predict(X_input)[0]
    return jsonify({"response": prediction})
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
