# Movie Review Sentiment Analysis (Naïve Bayes)

A web-based sentiment classifier built from scratch using a Multinomial Naïve Bayes algorithm.  
Enter any movie review—get an instant classification (Positive / Negative) with probability scores.


#### Project Overview

Algorithm: Multinomial Naïve Bayes

Preprocessing:

Lowercasing, HTML and punctuation removal

Tokenization (split on spaces)

Stopword removal using a custom list

#### Training Pipeline:

Calculates class priors and per-word likelihoods with Laplace smoothing (α = 1.0)

Uses log-space computations for numerical stability

#### Web Interface:

Clean, minimal HTML form to enter reviews

Result page shows the final sentiment (colored Positive / Negative), probability scores, and a "Try Again" button



#### Getting Started

1. Clone the repository
```
git clone https://github.com/priscillanzula/Movie-Review-Analysis_Naive_bayes.git
cd Movie-Review-Analysis_Naive_bayes
```
2. Install dependencies
   
```
pip install -r requirements.txt
```

3. Run the app
```
python app.py
```

View in browser

Navigate to http://127.0.0.1:5000/form, input your review, and see the sentiment result.

#### Demo




#### Why This Project Matters

Built from scratch – no black-box libraries, you can learn how Naïve Bayes works under the hood.

Compact and interpretable – easy to understand and modify.

Educational value – ideal baseline for text classification tasks.

Lightweight & fast – suitable for deployment as a standalone web app.

#### Future Improvements

Preserve negations (not, never) in preprocessing

Add n-gram support (e.g., bigrams such as "not good")

Implement feature selection or dimensionality reduction

Introduce hyperparameter tuning or cross-validation

Containerize using Docker and deploy on cloud platforms (e.g., Render, Heroku)

#### License & Contact

© 2025 Priscilla Nzula 
