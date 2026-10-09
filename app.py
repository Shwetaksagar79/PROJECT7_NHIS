from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("disaster_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        tweet = request.form["tweet"]
        vectorized = tfidf.transform([tweet])
        prediction = model.predict(vectorized)[0]
        result = "🚨 Disaster Tweet" if prediction == 1 else "✅ Not Disaster"
        return render_template("index.html", result=result, tweet=tweet)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
