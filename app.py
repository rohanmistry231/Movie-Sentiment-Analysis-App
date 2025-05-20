import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import io

# Function to predict sentiment
def predict_sentiment(review, tokenizer, model):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence, verbose=0)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    return sentiment, confidence

# Set page configuration (this sets the browser tab title and icon)
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",  # Browser tab title
    page_icon="🎬",  # Browser tab icon
    layout="wide"
)

# Cache model and tokenizer loading
@st.cache_resource
def load_model_and_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        model = tf.keras.models.load_model('model.h5')
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

# Home page
def home_page():
    st.title("Welcome to Movie Review Sentiment Analysis 🎬")
    st.markdown("""
    ### About This Application
    This application allows you to analyze movie reviews and predict whether they express positive or negative sentiment. Powered by a deep learning model trained on the IMDB dataset using LSTM, it provides accurate sentiment predictions with confidence scores.

    **Key Features:**
    - Analyze any movie review by entering text.
    - Receive instant predictions with sentiment (Positive/Negative) and confidence levels.
    - Try example reviews to see the model in action.
    - Download your analysis history as a Markdown file.
    - User-friendly interface with responsive design.

    ### How to Use This Application
    1. **Navigate to the Prediction Page**: Use the sidebar to select the "Prediction" option.
    2. **Enter a Review**: Type or paste a movie review in the text area.
    3. **Analyze Sentiment**: Click "Analyze Sentiment" to get the prediction.
    4. **Try Examples**: Use the sidebar buttons to test pre-loaded example reviews.
    5. **Download History**: Save your analysis history as a Markdown file.

    ### Get Started
    Select the **Prediction** page from the sidebar to start analyzing movie reviews!
    """)

# Prediction page
def prediction_page():
    st.title("Movie Review Sentiment Prediction")
    
    # Initialize session state for history
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()
    if tokenizer is None or model is None:
        st.error("Failed to load model or tokenizer. Please check the files and try again.")
        return

    # Input section
    st.header("Review Input")
    with st.form(key='review_form'):
        review_text = st.text_area("Enter your movie review:", placeholder="e.g., This movie was fantastic!", height=150)
        submit_button = st.form_submit_button(label="Analyze Sentiment")

    # Prediction
    if submit_button and review_text:
        with st.spinner("Analyzing..."):
            sentiment, confidence = predict_sentiment(review_text, tokenizer, model)
            
            st.subheader("Analysis Result:")
            if sentiment == "Positive":
                st.success(f"🎉 Sentiment: {sentiment}")
            else:
                st.error(f"😔 Sentiment: {sentiment}")
                
            st.write(f"Confidence: {confidence:.2%}")
            
            st.subheader("Your Review:")
            st.write(review_text)
            
            # Add to history
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p IST, %A, %B %d, %Y")
            st.session_state.analysis_history.append({
                "review": review_text,
                "sentiment": sentiment,
                "confidence": f"{confidence:.2%}",
                "timestamp": current_time
            })

    # Example reviews in sidebar
    st.sidebar.header("Try Example Reviews")
    example_reviews = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Terrible waste of time, poor direction and boring plot.",
        "It was okay, not great but watchable."
    ]
    
    for i, example in enumerate(example_reviews, 1):
        if st.sidebar.button(f"Example {i}"):
            sentiment, confidence = predict_sentiment(example, tokenizer, model)
            
            st.subheader("Example Analysis Result:")
            if sentiment == "Positive":
                st.success(f"🎉 Sentiment: {sentiment}")
            else:
                st.error(f"😔 Sentiment: {sentiment}")
                
            st.write(f"Confidence: {confidence:.2%}")
            st.subheader("Example Review:")
            st.write(example)
            
            # Add to history
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p IST, %A, %B %d, %Y")
            st.session_state.analysis_history.append({
                "review": example,
                "sentiment": sentiment,
                "confidence": f"{confidence:.2%}",
                "timestamp": current_time
            })

    # History section
    with st.expander("View Analysis History", expanded=False):
        if st.session_state.analysis_history:
            for entry in st.session_state.analysis_history:
                st.write(f"**Review** (Analyzed on {entry['timestamp']}):")
                st.write(entry['review'])
                st.write(f"**Sentiment**: {entry['sentiment']}")
                st.write(f"**Confidence**: {entry['confidence']}")
                st.divider()
        else:
            st.write("No analysis history available.")

    # Format history as Markdown for download
    def format_history_as_markdown():
        if not st.session_state.analysis_history:
            return "# Analysis History\n\nNo analysis history available."
        
        markdown_content = "# Analysis History\n\n"
        for entry in st.session_state.analysis_history:
            markdown_content += f"## Review (Analyzed on {entry['timestamp']})\n"
            markdown_content += f"{entry['review']}\n\n"
            markdown_content += f"## Result\n"
            markdown_content += f"- **Sentiment**: {entry['sentiment']}\n"
            markdown_content += f"- **Confidence**: {entry['confidence']}\n\n"
            markdown_content += "---\n\n"
        
        return markdown_content

    # Buttons layout (Clear History and Download History)
    st.markdown(
        """
        <style>
        .button-container {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        div.stButton > button.clear-history-button {
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
        }
        div.stButton > button.clear-history-button:hover {
            background-color: #e04343;
        }
        div.stButton > button.download-history-button {
            background-color: #4b9bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
        }
        div.stButton > button.download-history-button:hover {
            background-color: #437de0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            clear_history_button = st.button("Clear History", key="clear_history_button", help="Clear the analysis history", type="primary")
        with col2:
            markdown_content = format_history_as_markdown()
            buffer = io.StringIO()
            buffer.write(markdown_content)
            st.download_button(
                label="Download History",
                data=buffer.getvalue(),
                file_name="analysis_history.md",
                mime="text/markdown",
                key="download_history_button",
                help="Download the analysis history as a Markdown file",
                type="primary"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Handle Clear History button
    if clear_history_button:
        st.session_state.analysis_history = []
        st.rerun()

# Main app
def main():
    with st.sidebar:
        selected = option_menu(
            "Movie Sentiment Analysis",
            ["Home", "Prediction"],
            menu_icon="film",
            icons=["house", "bar-chart"],
            default_index=0
        )

    if selected == "Home":
        home_page()
    elif selected == "Prediction":
        prediction_page()

if __name__ == "__main__":
    main()