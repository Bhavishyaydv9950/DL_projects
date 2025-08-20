from flask import Flask, render_template, request
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Step 1: Prepare Data
# -----------------------------
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(parts[1])
    return texts, labels

train_texts, train_labels = load_data(r"C:\Users\dell\Desktop\DATA SICENCE\1-july-2025-Data-Science\deep learning\RNN\data\train.txt")
test_texts, test_labels   = load_data(r"C:\Users\dell\Desktop\DATA SICENCE\1-july-2025-Data-Science\deep learning\RNN\data\test.txt")


# -----------------------------
# Step 2: Train Model
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

model = LogisticRegression(max_iter=200)
model.fit(X_train, train_labels)

# Save model & vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# -----------------------------
# Step 3: Flask App
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["user_text"]
    vector = vectorizer.transform([user_text])
    prediction = model.predict(vector)[0]
    return render_template("index.html", prediction=prediction, user_text=user_text)

if __name__ == "__main__":
    app.run(debug=True)
