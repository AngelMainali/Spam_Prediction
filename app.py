
"""
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

"""



# # working fine with LIME 
# import streamlit as st
# import pandas as pd
# import pickle
# import string
# import re
# import os
# import nltk
# import numpy as np
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from lime.lime_text import LimeTextExplainer
# import streamlit.components.v1 as components

# # ---------------- Setup ----------------
# nltk.download('stopwords')
# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# st.set_page_config(page_title="📩 SMS Spam Detector with XAI", layout="centered")
# st.title("📩 SMS Spam Detector with Explainable AI (LIME)")

# # ---------------- Helper Function ----------------
# def transform_text(text):
#     text = str(text).lower()
#     text = re.findall(r'\b\w+\b', text)
#     text = [i for i in text if i not in stop_words and i not in string.punctuation]
#     text = [ps.stem(i) for i in text]
#     return " ".join(text)

# # ---------------- Load Model ----------------
# MODEL_PATH = "C:/Users/Dell/Downloads/SpamEmailDetection/model.pkl"
# VECTORIZER_PATH = "C:/Users/Dell/Downloads/SpamEmailDetection/vectorizer.pkl"

# if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
#     st.warning("⚠️ Model or vectorizer not found. Please train and save them first.")
# else:
#     model = pickle.load(open(MODEL_PATH, 'rb'))
#     tfidf = pickle.load(open(VECTORIZER_PATH, 'rb'))

#     # ---------------- UI ----------------
#     user_input = st.text_area("✉️ Enter your SMS message:")

#     if st.button("Predict"):
#         if user_input.strip() == "":
#             st.warning("⚠️ Please enter a message first!")
#         else:
#             # Preprocess input
#             transformed_msg = transform_text(user_input)
#             vector_input = tfidf.transform([transformed_msg])
#             prediction = model.predict(vector_input)[0]
#             pred_proba = model.predict_proba(vector_input)[0]

#             if prediction == 1:
#                 st.error(f"🚨 This message is **Spam!** (Confidence: {pred_proba[1]*100:.2f}%)")
#             else:
#                 st.success(f"✅ This message is **Not Spam (Ham)** (Confidence: {pred_proba[0]*100:.2f}%)")

#             st.subheader("🧠 Model Explanation")

#             # ---------------- LIME EXPLANATION ----------------
#             class_names = ['ham', 'spam']

#             def predict_proba(texts):
#                 X = tfidf.transform([transform_text(t) for t in texts])
#                 return model.predict_proba(X)

#             lime_explainer = LimeTextExplainer(class_names=class_names)
#             lime_exp = lime_explainer.explain_instance(user_input, predict_proba, num_features=10)

#             # Wrap LIME HTML in a white container for dark theme
#             lime_html = lime_exp.as_html()
#             lime_html = f"""
#             <div style="background-color:white; padding:15px; border-radius:10px;">
#             {lime_html}
#             </div>
#            """

#             st.markdown("**LIME Explanation:**")
#             components.html(lime_html, height=600)






import streamlit as st
import pandas as pd
import pickle
import string
import re
import os
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from lime.lime_text import LimeTextExplainer
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# ---------------- Setup ----------------
nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="📩 SMS Spam Detector with XAI", layout="centered")
st.title("📩 SMS Spam Detector with Explainable AI (LIME + SHAP)")

# ---------------- Helper Function ----------------
def transform_text(text):
    text = str(text).lower()
    text = re.findall(r'\b\w+\b', text)
    text = [i for i in text if i not in stop_words and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# ---------------- Load Model ----------------
MODEL_PATH = "C:/Users/Dell/Downloads/SpamEmailDetection/model.pkl"
VECTORIZER_PATH = "C:/Users/Dell/Downloads/SpamEmailDetection/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.warning("⚠️ Model or vectorizer not found. Please train and save them first.")
else:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    tfidf = pickle.load(open(VECTORIZER_PATH, 'rb'))

    # ---------------- UI ----------------
    user_input = st.text_area("✉️ Enter your SMS message:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a message first!")
        else:
            # Preprocess input
            transformed_msg = transform_text(user_input)
            vector_input = tfidf.transform([transformed_msg])
            prediction = model.predict(vector_input)[0]
            pred_proba = model.predict_proba(vector_input)[0]

            if prediction == 1:
                st.error(f"🚨 This message is **Spam!** (Confidence: {pred_proba[1]*100:.2f}%)")
            else:
                st.success(f"✅ This message is **Not Spam (Ham)** (Confidence: {pred_proba[0]*100:.2f}%)")

            st.subheader("🧠 Model Explanation")

            # ---------------- LIME EXPLANATION ----------------
            class_names = ['ham', 'spam']

            def predict_proba(texts):
                X = tfidf.transform([transform_text(t) for t in texts])
                return model.predict_proba(X)

            lime_explainer = LimeTextExplainer(class_names=class_names)
            lime_exp = lime_explainer.explain_instance(user_input, predict_proba, num_features=10)

            # Wrap LIME HTML in a white container for dark theme
            lime_html = lime_exp.as_html()
            lime_html = f"""
            <div style="background-color:white; padding:15px; border-radius:10px;">
            {lime_html}
            </div>
            """
            st.markdown("**LIME Explanation:**")
            components.html(lime_html, height=600)

            # ---------------- SHAP EXPLANATION ----------------
            st.subheader("🧠 SHAP Explanation (Local)")

            # Wrap model + TF-IDF as callable function
            def model_predict(texts):
                X = tfidf.transform([transform_text(t) for t in texts])
                return model.predict_proba(X)

            # SHAP Text Explainer
            explainer_text = shap.Explainer(model_predict, masker=shap.maskers.Text())
            shap_values_text = explainer_text([user_input])

          # ---------------- Word-level contributions ----------------
            st.markdown("**Word-level contributions:**")

            # Generate SHAP HTML for the text plot
            shap_html = shap.plots.text(shap_values_text[0], display=False)  # returns HTML string

            # Wrap in a white container for readability
            shap_html_wrapped = f"""
            <div style="background-color:white; padding:15px; border-radius:10px;">
                {shap_html}
            </div>
            """

            # Render in Streamlit
            components.html(shap_html_wrapped, height=300)

           # ---------------- Force plot for spam class (Static Matplotlib) ----------------
   
            shap_values_instance = shap_values_text.values[0][:, 1]  # spam class
            words = shap_values_text.data[0]

            force_expl = shap.Explanation(
                values=shap_values_instance,
                base_values=shap_values_text.base_values[0][1],
                data=words,
                feature_names=words
            )

            st.markdown("**Force Plot for Spam Class :**")

            # Let SHAP generate its figure internally
            fig_force = shap.plots.force(force_expl, matplotlib=True, show=False)  # returns the figure

            # Pass the returned figure explicitly to Streamlit
            st.pyplot(fig_force)
            plt.close(fig_force) # Close the figure to avoid overlapping plots


            # ---------------- Summary plot ----------------
            shap_values_for_plot = shap_values_instance.reshape(1, -1)
            st.markdown("**Summary Plot (Single Message):**")
            fig_summary, ax = plt.subplots(figsize=(10, 4))
            shap.summary_plot(
                shap_values_for_plot,
                features=[words],
                feature_names=words,
                plot_type='bar',
                show=False
            )
            st.pyplot(fig_summary)
            plt.close(fig_summary)

           

                    # ---------------- Global SHAP ----------------
            
           # ---------------- SHAP GLOBAL EXPLANATION ----------------
            st.subheader("🧠 SHAP Global Explanation (Entire Dataset)")

            DATASET_PATH = "C:/Users/Dell/Downloads/SpamEmailDetection/spam.csv"

            if os.path.exists(DATASET_PATH):
                try:
                    data_df = pd.read_csv(DATASET_PATH, encoding='latin-1')
                    
                    # Use only a subset for performance (e.g., first 500 samples)
                    sample_size = min(500, len(data_df))
                    data_sample = data_df.head(sample_size)
                    
                    texts = data_sample['v2'].apply(transform_text)
                    X = tfidf.transform(texts)
                    X_dense = X.toarray()
                    
                    # Use small samples to avoid memory issues
                    background_size = 100
                    explain_size = 300
                    
                    with st.spinner('Computing global SHAP explanations... (This may take a minute)'):
                        # Convert to dense for the small subset only
                        X_dense_small = X_dense[:background_size + explain_size]
                        background = X_dense_small[:background_size]
                        explain_data = X_dense_small[background_size:background_size + explain_size]
                        
                        explainer_global = shap.KernelExplainer(model.predict_proba, background)
                        shap_values_global = explainer_global.shap_values(explain_data)
                    
                    # Handle SHAP values dimensions
                    if isinstance(shap_values_global, list):
                        shap_values_to_plot = shap_values_global[1]  # Use class 1 (spam)
                    else:
                        if len(shap_values_global.shape) == 3:
                            shap_values_to_plot = shap_values_global[:, :, 1]  # Class 1 for binary
                        else:
                            shap_values_to_plot = shap_values_global
                    
                    # Calculate mean absolute SHAP values
                    mean_abs_shap = np.mean(np.abs(shap_values_to_plot), axis=0)
                    feature_names = tfidf.get_feature_names_out()
                    
                    # Get top 15 features
                    top_indices = np.argsort(mean_abs_shap)[-15:][::-1]
                    top_features = [feature_names[i] for i in top_indices]
                    top_scores = mean_abs_shap[top_indices]
                    
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    y_pos = np.arange(len(top_features))
                    
                    bars = ax.barh(y_pos, top_scores, color='#1f77b4')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_features)
                    ax.set_xlabel('Mean |SHAP value|')
                    ax.set_title('Global SHAP Feature Importance - Top 15 Features')
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                                f'{width:.4f}', ha='left', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.success(f"Global SHAP analysis completed on {sample_size} samples!")
                    
                except Exception as e:
                    st.error(f"Global SHAP computation failed: {str(e)}")
                    st.info("This might be due to memory limitations. Try with a smaller dataset.")
            else:
                st.warning("Dataset not found for global explanations. Please ensure spam.csv is available.")