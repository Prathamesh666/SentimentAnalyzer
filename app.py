import streamlit as st
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Configure page style
st.set_page_config(page_title="NLP Sentiment Analyzer", page_icon="🎬", layout="centered")

# Ensure NLTK resources are available and loaded
@st.cache_resource
def load_nltk_resources():
    nltk.download('movie_reviews')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    return set(stopwords.words('english'))

stop_words = load_nltk_resources()

def clean_text(text):
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(cleaned_tokens)

# Cache model training so it only executes once on app load
@st.cache_resource
def train_and_initialize_models():
    # 1. Load Data
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append({
                'text': movie_reviews.raw(fileid),
                'sentiment': category
            })
    df_data = pd.DataFrame(documents)
    df_data['cleaned_text'] = df_data['text'].apply(clean_text)
    df_data['label'] = df_data['sentiment'].map({'pos': 1, 'neg': 0})

    # 2. Train-Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df_data['cleaned_text'],
        df_data['label'],
        test_size=0.2,
        random_state=42,
        stratify=df_data['label']
    )

    # 3. Fit TF-IDF Vectorizer & Train Classical ML Models
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # 4. Tokenize and Build LSTM Network
    max_words = 10000
    max_len = 200
    embedding_dim = 128

    lstm_tokenizer = Tokenizer(num_words=max_words)
    lstm_tokenizer.fit_on_texts(X_train_raw)
    X_train_seq = pad_sequences(lstm_tokenizer.texts_to_sequences(X_train_raw), maxlen=max_len)

    lstm_model = Sequential([
        Embedding(max_words, embedding_dim),
        Dropout(0.3),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train LSTM briefly for demo purposes
    lstm_model.fit(X_train_seq, y_train, epochs=3, batch_size=64, verbose=0)

    return vectorizer, lr_model, nb_model, lstm_tokenizer, lstm_model

with st.spinner("Training sentiment models from scratch inside the app container... Please wait!"):
    vectorizer, lr_model, nb_model, lstm_tokenizer, lstm_model = train_and_initialize_models()

# CSS for custom animated metrics cards
custom_css = """
<style>
.sentiment-card {
    border-radius: 12px 12px 0px 0px;
    padding: 20px;
    margin-top: 10px;
    margin-bottom: 0px;
    text-align: center;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    color: #ffffff;
}
.pos-card {
    background-color: #28a745;
}
.neu-card {
    background-color: #ffc107;
    color: #1a1a1a;
}
.neg-card {
    background-color: #dc3545;
}
.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.card-value {
    font-size: 2rem;
    font-weight: bold;
}
.card-container {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 15px;
}
.card-container:hover {
    transform: translateY(-5px) scale(1.03);
}
.progress-container {
    background-color: #e0e0e0;
    border-radius: 0px 0px 12px 12px;
    width: 100%;
    height: 12px;
    box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 0px 0px 0px 12px;
    transition: width 0.5s ease-in-out;
}
.pos-bar {
    background-color: #1e7e34;
}
.neu-bar {
    background-color: #d39e00;
}
.neg-bar {
    background-color: #bd2130;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True) 

# Main UI layout
st.title("🎬 Robust Sentiment Analyzer")
st.write("Analyze sentiments expressed dynamically with our in-memory trained NLP models.")
st.markdown("---")

# Sidebar / Dropdown for Model Selection
selected_model_name = st.selectbox(
    "Choose Sentiment Prediction Model:",
    ["Logistic Regression", "Naive Bayes", "LSTM (Long Short-Term Memory)"]
)

# Input field
user_input = st.text_area(
    "Enter a comment/review/feedback to classify:",
    "The cinematography was absolutely stunning and the acting was top-notch, though the plot felt a bit rushed towards the end. Overall, a highly enjoyable experience!"
)

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("Please write some text to analyze!")
    else:
        cleaned = clean_text(user_input)
        st.subheader("Preprocessing Result")
        st.info(f"**Cleaned Text:** {cleaned if cleaned else '[No words left after filtering]'}")

        st.subheader(f"Sentiment Dashboard ({selected_model_name})")
        if not cleaned.strip():
            st.error("Cannot make prediction because no vocabulary words were left after cleaning!")
        else:
            # Fetch probability of being Positive (class 1)
            if selected_model_name == "Logistic Regression":
                vectorized_input = vectorizer.transform([cleaned])
                prob_pos = lr_model.predict_proba(vectorized_input)[0][1]
            elif selected_model_name == "Naive Bayes":
                vectorized_input = vectorizer.transform([cleaned])
                prob_pos = nb_model.predict_proba(vectorized_input)[0][1]
            elif selected_model_name == "LSTM (Long Short-Term Memory)":
                seq = lstm_tokenizer.texts_to_sequences([cleaned])
                padded_seq = pad_sequences(seq, maxlen=200)
                prob_pos = float(lstm_model.predict(padded_seq, verbose=0)[0][0])

            # Derive 3-way distribution (Positive, Negative, Neutral)
            neutral_margin = 0.08
            lower_bound = 0.5 - neutral_margin
            upper_bound = 0.5 + neutral_margin

            if prob_pos > upper_bound:
                neutral_pct = (1.0 - prob_pos) * 0.4
                pos_pct = prob_pos
                neg_pct = 1.0 - pos_pct - neutral_pct
                sentiment_label = "Positive"
                status_color = st.success
                st.balloons()
            elif prob_pos < lower_bound:
                neutral_pct = prob_pos * 0.4
                neg_pct = 1.0 - prob_pos
                pos_pct = 1.0 - neg_pct - neutral_pct
                sentiment_label = "Negative"
                status_color = st.error
                st.snow()
            else:
                neutral_pct = 0.6 + (0.1 * (1.0 - abs(prob_pos - 0.5)/neutral_margin))
                remaining = 1.0 - neutral_pct
                pos_pct = remaining * (prob_pos / (prob_pos + (1.0 - prob_pos)))
                neg_pct = remaining - pos_pct
                sentiment_label = "Neutral"
                status_color = st.warning
                st.snow()
                st.balloons()

            # Normalize percentages
            total = pos_pct + neg_pct + neutral_pct
            pos_pct /= total
            neg_pct /= total
            neutral_pct /= total

            status_color(f"### **Overall Sentiment:** {sentiment_label}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="card-container">
                    <div class="sentiment-card pos-card">
                        <div class="card-title">🟢 POSITIVE</div>
                        <div class="card-value">{pos_pct:.1%}</div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar-fill pos-bar" style="width: {pos_pct*100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="card-container">
                    <div class="sentiment-card neu-card">
                        <div class="card-title">🟡 NEUTRAL</div>
                        <div class="card-value">{neutral_pct:.1%}</div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar-fill neu-bar" style="width: {neutral_pct*100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="card-container">
                    <div class="sentiment-card neg-card">
                        <div class="card-title">🔴 NEGATIVE</div>
                        <div class="card-value">{neg_pct:.1%}</div>
                </div>
                    <div class="progress-container">
                        <div class="progress-bar-fill neg-bar" style="width: {neg_pct*100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
