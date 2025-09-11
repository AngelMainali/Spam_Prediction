import streamlit as st
import pandas as pd
import pickle
import string
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import nltk

# ---------------- Setup ----------------
nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="SMS Spam Detection & Analysis", layout="wide")
st.title("📩 SMS Spam Detection & Analysis")

# ---------------- Helpers ----------------
def transform_text(text):
    text = str(text).lower()
    text = re.findall(r'\b\w+\b', text)
    text = [i for i in text if i not in stop_words and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

def detect_text_column(df):
    # If known column name exists
    if 'v2' in df.columns:
        return 'v2'
    # Try to find the column with the most string-like entries
    text_col = None
    max_text_ratio = 0
    for col in df.columns:
        text_ratio = df[col].apply(lambda x: isinstance(x, str)).mean()
        if text_ratio > max_text_ratio:
            max_text_ratio = text_ratio
            text_col = col
    return text_col if max_text_ratio > 0.5 else None

# ---------------- Load Model ----------------
MODEL_PATH = "C:/Users/Dell/Downloads/SpamEmailDetection/model.pkl"
VECTORIZER_PATH = "C:/Users/Dell/Downloads/SpamEmailDetection/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.warning("⚠️ Model or vectorizer not found. Tab 1 will not work until you train and save them.")
else:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    tfidf = pickle.load(open(VECTORIZER_PATH, 'rb'))

tab1, = st.tabs(["📩 Predict SMS Spam & Analyze"])

with tab1:

    user_input = st.text_area("Enter your SMS message to get **spam prediction** and **rich message analysis**.")

    if st.button("Predict & Analyze"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a message first!")
        else:
            # ---------------- Prediction ----------------
            transformed_msg = transform_text(user_input)
            vector_input = tfidf.transform([transformed_msg])
            prediction = model.predict(vector_input)[0]
            prediction_label = "⚠️ Spam" if prediction == 1 else "✅ Ham (Not Spam)"
            st.markdown(f"### Prediction: **{prediction_label}**")

            # ---------------- Analysis ----------------
            import matplotlib.pyplot as plt
            import seaborn as sns
            from collections import Counter
            from textblob import TextBlob
            import textstat
            from wordcloud import WordCloud
            import pandas as pd
            import numpy as np

            words = [w for w in transformed_msg.split() if w.isalpha()]
            word_freq = Counter(words)
            char_freq = Counter(user_input)
            word_lengths = [len(w) for w in words]

            num_chars = len(user_input)
            num_words = len(user_input.split())
            num_sentences = user_input.count('.') + user_input.count('!') + user_input.count('?')
            num_digits = sum(c.isdigit() for c in user_input)
            num_uppercase = sum(1 for w in user_input.split() if w.isupper())
            num_exclaims = user_input.count('!')
            num_questions = user_input.count('?')

            stop_words = set(nltk.corpus.stopwords.words('english'))
            num_stopwords = sum(1 for w in user_input.lower().split() if w in stop_words)
            num_content_words = num_words - num_stopwords

            blob = TextBlob(user_input)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            readability = textstat.flesch_reading_ease(user_input)
            lexical_div = len(set(words))/len(words) if len(words) > 0 else 0
            avg_syllables = np.mean([textstat.syllable_count(w) for w in words]) if len(words) > 0 else 0

            # ---------------- Metrics Display ----------------
            st.markdown("### 🧮 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📝 Characters", num_chars)
            col2.metric("🔤 Words", num_words)
            col3.metric("📏 Sentences", num_sentences)
            col4.metric("📚 Readability", f"{readability:.1f}")


            col5, col6, col7, col8 = st.columns(4)
            col5.metric("💖 Polarity", f"{polarity:.2f}")
            col6.metric("🧐 Subjectivity", f"{subjectivity:.2f}")
            col7.metric("🔢 Numbers", num_digits)
            col8.metric("🔠 Uppercase Words", num_uppercase)


            col9, col10, col11, col12 = st.columns(4)
            col9.metric("❗ Exclamations", num_exclaims)
            col10.metric("❓ Questions", num_questions)
            col11.metric("📊 Stopwords", num_stopwords)
            col12.metric("✏️ Content Words", num_content_words)

            col13, col14 = st.columns(2)
            col13.metric("🔤 Avg Syllables/Word", f"{avg_syllables:.2f}")
            col14.metric("🧩 Lexical Diversity", f"{lexical_div:.2f}")

            st.markdown("---")
            st.markdown("### 📊 Visual Insights")

            # ---------------- Visualizations Side-by-Side ----------------
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            col5, col6 = st.columns(2)

            # Word Frequency
            if len(word_freq) > 0:
                freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Count']).sort_values('Count', ascending=False)
                fig, ax = plt.subplots(figsize=(3,2))
                sns.barplot(x='Count', y='Word', data=freq_df.head(5), ax=ax, palette="viridis")
                ax.set_title('Top Words', fontsize=8)
                ax.tick_params(axis='both', labelsize=7)
                col1.pyplot(fig, use_container_width=True)

            # Character Frequency
            if len(char_freq) > 0:
                char_df = pd.DataFrame(char_freq.items(), columns=['Character','Count']).sort_values('Count', ascending=False)
                fig, ax = plt.subplots(figsize=(3,2))
                sns.barplot(x='Count', y='Character', data=char_df.head(10), ax=ax, palette="magma")
                ax.set_title('Character Distribution', fontsize=8)
                ax.tick_params(axis='both', labelsize=7)
                col2.pyplot(fig, use_container_width=True)

            # Word Length Distribution
            fig, ax = plt.subplots(figsize=(3,2))
            sns.histplot(word_lengths, bins=5, kde=True, color="Blue", ax=ax)
            ax.axvline(np.mean(word_lengths), color='red', linestyle='--', label='Mean')
            ax.set_xlabel("Word Length", fontsize=7)
            ax.set_ylabel("Frequency", fontsize=7)
            ax.set_title("Word Lengths", fontsize=8)
            ax.legend(fontsize=6)
            col3.pyplot(fig, use_container_width=True)

            # Stopwords vs Content Words Donut
            fig, ax = plt.subplots(figsize=(3,2.5))
            ax.pie([num_stopwords, num_content_words], labels=["Stopwords","Content Words"], autopct='%1.0f%%',
                   colors=['#FF9999','#99FF99'], wedgeprops={'width':0.5})
            ax.set_title("Stopwords vs Content", fontsize=8)
            col4.pyplot(fig, use_container_width=True)

            # Exclamations vs Questions
            fig, ax = plt.subplots(figsize=(3,2))
            sns.barplot(x=['Exclamations','Questions'], y=[num_exclaims,num_questions], palette="pastel", ax=ax)
            ax.set_title("Punctuation Count", fontsize=8)
            ax.set_ylabel("Count", fontsize=7)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            col5.pyplot(fig, use_container_width=True)

            # Word Cloud
            if len(words) > 0:
                wc = WordCloud(width=250, height=120, background_color='white').generate(' '.join(words))
                fig, ax = plt.subplots(figsize=(3,2))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title("Word Cloud", fontsize=8)
                col6.pyplot(fig, use_container_width=True)

            # ---------------- Disclaimer ----------------
            st.markdown("---")
            st.info("**Disclaimer:** This model predicts spam/ham probabilistically. Analysis shows sentiment, readability, punctuation, and word patterns. Always verify messages asking for personal info, money, or codes.")

