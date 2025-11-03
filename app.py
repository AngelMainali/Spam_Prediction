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
st.title("📩 Explainable SMS Spam Detector")

# ---------------- Helper Function ----------------
def transform_text(text):
    text = str(text).lower()
    text = re.findall(r'\b\w+\b', text)
    text = [i for i in text if i not in stop_words and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# ---------------- Load Model ----------------
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.warning("⚠️ Model or vectorizer not found. Please train and save them first.")
else:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    tfidf = pickle.load(open(VECTORIZER_PATH, 'rb'))

    # ---------------- Model Prediction Function ----------------
    def model_predict(texts):
        X = tfidf.transform([transform_text(t) for t in texts])
        return model.predict_proba(X)

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

            # ---------------- LIME LOCAL EXPLANATION ----------------
            st.markdown("### 📊 LIME Explanation")
            
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
            
            components.html(lime_html, height=600)

            # ---------------- SHAP LOCAL EXPLANATION ----------------
            st.subheader("🧠 SHAP Local Explanation")

            # SHAP Text Explainer
            explainer_text = shap.Explainer(model_predict, masker=shap.maskers.Text())
            shap_values_text = explainer_text([user_input])
            shap_values_instance = shap_values_text.values[0][:, 1]  # spam class
            words = shap_values_text.data[0]

             # ---------------- Summary plot ----------------
            shap_values_for_plot = shap_values_instance.reshape(1, -1)
            st.markdown("**Summary Plot:**")
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

                        # ---------------- Waterfall Plot ----------------
            st.markdown("**Waterfall Plot:**")

            # Create waterfall plot for spam class


            # Create waterfall plot
            fig_waterfall, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values_text[0][:, 1], show=False)
            plt.tight_layout()
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)


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



            

            st.markdown("**Force Plot for Spam Class:**")

            # Let SHAP generate its figure internally
            fig_force = shap.plots.force(force_expl, matplotlib=True, show=False)  # returns the figure

            # Pass the returned figure explicitly to Streamlit
            st.pyplot(fig_force)
            plt.close(fig_force) # Close the figure to avoid overlapping plots

           

            # ---------------- HYBRID EXPLANATION (Built-in) ----------------
            st.subheader("🔗 Hybrid Explanation (LIME + SHAP)")

            

            # Get LIME features
            lime_features = lime_exp.as_list()

            # Get SHAP values
            shap_values_instance = shap_values_text.values[0][:, 1]  # spam class
            words = shap_values_text.data[0]

            # Combine LIME and SHAP explanations
            hybrid_data = []
            for feature, lime_weight in lime_features:
                # Find matching SHAP value
                shap_weight = 0
                for i, word in enumerate(words):
                    if feature in word or word in feature:
                        shap_weight = shap_values_instance[i]
                        break
                
                # Calculate hybrid score (weighted average)
                hybrid_score = (lime_weight + shap_weight) / 2
                hybrid_data.append({
                    'feature': feature,
                    'lime_weight': lime_weight,
                    'shap_weight': shap_weight,
                    'hybrid_weight': hybrid_score
                })

            # Sort by absolute hybrid weight
            hybrid_data_sorted = sorted(hybrid_data, key=lambda x: abs(x['hybrid_weight']), reverse=True)

            # Create hybrid feature importance plot
            features = [item['feature'] for item in hybrid_data_sorted]
            lime_weights = [item['lime_weight'] for item in hybrid_data_sorted]
            shap_weights = [item['shap_weight'] for item in hybrid_data_sorted]
            hybrid_weights = [item['hybrid_weight'] for item in hybrid_data_sorted]




                        # Display detailed comparison table
            st.markdown("**Detailed Method Comparison:**")

            comparison_df = pd.DataFrame(hybrid_data_sorted)
            comparison_df['impact_type'] = comparison_df['hybrid_weight'].apply(
                lambda x: '🟥 SPAM Indicator' if x > 0 else '🟩 HAM Indicator'
            )

            # Format the display
            display_df = comparison_df[['feature', 'impact_type', 'lime_weight', 'shap_weight', 'hybrid_weight']]
            display_df.columns = ['Feature', 'Impact Type', 'LIME Weight', 'SHAP Weight', 'Hybrid Weight']

            st.dataframe(
                display_df.style.background_gradient(
                    subset=['LIME Weight', 'SHAP Weight', 'Hybrid Weight'], 
                    cmap='RdYlGn_r'
                ).format({
                    'LIME Weight': '{:.4f}',
                    'SHAP Weight': '{:.4f}', 
                    'Hybrid Weight': '{:.4f}'
                }),
                use_container_width=True
            )

            # Plot 1: Stacked Bar Chart (LIME + SHAP contributions)
            st.markdown("**Stacked Bar Chart - LIME vs SHAP Contributions:**")

            fig_stacked, ax = plt.subplots(figsize=(12, 8))
            y_pos = np.arange(len(features))

            # Create stacked bars
            bars_lime = ax.barh(y_pos, lime_weights, color='blue', alpha=0.7, label='LIME Contribution')
            bars_shap = ax.barh(y_pos, shap_weights, left=lime_weights, color='orange', alpha=0.7, label='SHAP Contribution')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance Score')
            ax.set_title('Stacked Bar Chart: LIME vs SHAP Contributions')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.legend()

            # Add value labels for total (LIME + SHAP)
            for i, (lime_val, shap_val) in enumerate(zip(lime_weights, shap_weights)):
                total = lime_val + shap_val
                if abs(total) > 0.001:  # Only label if significant
                    ax.text(total + (0.01 if total >= 0 else -0.01), i, 
                            f'{total:.3f}', ha='left' if total >= 0 else 'right', 
                            va='center', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig_stacked)
            plt.close(fig_stacked)

            # Plot 2: Hybrid Feature Importance (Individual bars)
            st.markdown("**Hybrid Feature Importance:**")

            fig_hybrid, ax = plt.subplots(figsize=(12, 6))
            colors = ['red' if imp > 0 else 'green' for imp in hybrid_weights]

            bars = ax.barh(y_pos, hybrid_weights, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Hybrid Importance Score')
            ax.set_title('Hybrid Feature Importance (LIME + SHAP Combined)')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01 if width >= 0 else width - 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left' if width >= 0 else 'right', 
                        va='center', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig_hybrid)
            plt.close(fig_hybrid)


            st.success("✅ Hybrid explanation completed! Combines LIME and SHAP for more robust feature importance.")

            # ---------------- GLOBAL SHAP EXPLANATION ----------------
            st.subheader("🌍 SHAP Global Explanation")

            DATASET_PATH = "spam.csv"

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
                    background_size = 20
                    explain_size = 30
                    
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
                    
                    # Create the bar plot
                    st.markdown("**Global SHAP Feature Importance - Top 15 Features:**")
                    fig_bar, ax = plt.subplots(figsize=(12, 8))
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
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)
                    
                    # ---------------- Beeswarm Plot ----------------
                    st.markdown("**SHAP Beeswarm Plot:**")
                    
                    # Create beeswarm plot
                    fig_beeswarm, ax = plt.subplots(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values_to_plot, 
                        features=explain_data,
                        feature_names=feature_names,
                        plot_type="violin",  # beeswarm plot
                        show=False,
                        max_display=15  # Show top 15 features
                    )
                    plt.tight_layout()
                    st.pyplot(fig_beeswarm)
                    plt.close(fig_beeswarm)
                    
                    # ---------------- Dependence Plots ----------------
                    st.markdown("**SHAP Dependence Plots:**")
                    
                    # Create dependence plots for top 3 features
                    top_3_indices = top_indices[:3]
                    top_3_features = [feature_names[i] for i in top_3_indices]
                    
                    # Create columns for the dependence plots
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (feature_idx, feature_name) in enumerate(zip(top_3_indices, top_3_features)):
                        # Select the appropriate column
                        with [col1, col2, col3][i]:
                            st.markdown(f"**Dependence Plot: '{feature_name}'**")
                            
                            # Create dependence plot
                            fig_dep, ax = plt.subplots(figsize=(8, 6))
                            shap.dependence_plot(
                                feature_idx,
                                shap_values_to_plot,
                                explain_data,
                                feature_names=feature_names,
                                ax=ax,
                                show=False,
                                alpha=0.7
                            )
                            ax.set_title(f"SHAP Dependence: '{feature_name}'")
                            plt.tight_layout()
                            st.pyplot(fig_dep)
                            plt.close(fig_dep)
                    
                    st.success(f"Global SHAP analysis completed on {sample_size} samples!")
                    
                except Exception as e:
                    st.error(f"Global SHAP computation failed: {str(e)}")
                    st.info("This might be due to memory limitations. Try with a smaller dataset.")
            else:
                st.warning("Dataset not found for global explanations. Please ensure spam.csv is available.")