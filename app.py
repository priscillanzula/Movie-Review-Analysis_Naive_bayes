from flask import Flask, request, jsonify
import json
import math
import re

# --- Load model ---
with open("nb_model.json", "r", encoding="utf-8") as f:
    MODEL = json.load(f)
MODEL["vocab_set"] = set(MODEL["vocab"])

STOPWORDS = set(["the", "and", "a", "an", "is", "it", "to", "i", "this", "that",
                "in", "of", "was", "do", "were", "for", "with", "as", "movie", "film"])

# --- Helper functions ---


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return tokens


def predict_proba(tokens, model):
    classes = list(model['class_priors'].keys())
    log_scores = {}
    for cls in classes:
        log_score = math.log(model['class_priors'][cls])
        denom = model['class_total_words'][cls] + \
            model['alpha'] * len(model['vocab'])
        for token in tokens:
            if token not in model['vocab_set']:
                continue
            count = model['word_counts'][cls].get(token, 0)
            log_score += math.log((count + model['alpha']) / denom)
        log_scores[cls] = log_score
    max_log = max(log_scores.values())
    exps = {c: math.exp(v - max_log) for c, v in log_scores.items()}
    total = sum(exps.values())
    return {c: exps[c]/total for c in exps}


def predict(text, model):
    tokens = clean_text(text)
    probs = predict_proba(tokens, model)
    label = max(probs, key=probs.get)
    return label, probs


# --- Flask app ---
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text'"}), 400
    label, probs = predict(data["text"], MODEL)
    return jsonify({"label": label, "probabilities": probs})


@app.route("/form")
def form():
    return """
    <html>
      <head>
        <title>Naive Bayes Sentiment Classifier</title>
        <style>
          body { font-family: Arial, sans-serif; background: #f8f9fa; text-align: center; padding: 50px; }
          h1 { color: #333; }
          form { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }
          input[type="text"] { width: 400px; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
          input[type="submit"] { padding: 10px 20px; border: none; border-radius: 5px; background: #007bff; color: white; font-size: 16px; cursor: pointer; }
          input[type="submit"]:hover { background: #0056b3; }
        </style>
      </head>
      <body>
        <h1>Movie Review Sentiment Analysis</h1>
        <form action="/analyze" method="post">
          <input name="text" placeholder="Enter your movie review here" size="60" required>
          <br><br>
          <input type="submit" value="Analyze Sentiment">
        </form>
      </body>
    </html>
    """


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text")
    if not text:
        return "Please enter some text!"

    label, probs = predict(text, MODEL)

    # Final decision with color
    if label == "pos":
        decision = "<h2 style='color:green'>Positive 😊</h2>"
    else:
        decision = "<h2 style='color:red'>Negative 😞</h2>"

    # Probabilities table
    probs_html = "<h3>Probabilities:</h3><ul style='list-style:none; padding:0;'>"
    for cls, p in probs.items():
        probs_html += f"<li><b>{cls}</b>: {p:.4f}</li>"
    probs_html += "</ul>"

    return f"""
    <html>
      <head>
        <title>Sentiment Result</title>
        <style>
          body {{ font-family: Arial, sans-serif; background: #f8f9fa; text-align: center; padding: 50px; }}
          .box {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); display: inline-block; }}
          a.button {{ display: inline-block; margin-top: 20px; padding: 10px 20px; border-radius: 5px; background: #6c757d; color: white; text-decoration: none; }}
          a.button:hover {{ background: #5a6268; }}
        </style>
      </head>
      <body>
        <div class="box">
          <h1>Sentiment Analysis Result</h1>
          {decision}
          {probs_html}
          <a href="/form" class="button">🔄 Analyze Another Review</a>
        </div>
      </body>
    </html>
    """


@app.route("/")
def home():
    return "Naive Bayes Sentiment Classifier is running!"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
