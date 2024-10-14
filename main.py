import streamlit as st
import re
from io import BytesIO
import pandas as pd
import pickle
import base64
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

# Load the necessary models and transformers
predictor = pickle.load(open(r"Models/trained_model.pkl", "rb"))
scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

# Stopwords and stemmer
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# HTML and CSS styling
# HTML and CSS styling
# HTML and CSS styling
st.markdown("""
    <style>
        /* Background */
        .main {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
        }

        /* Banner Image with proper fitting */
        .banner-img {
            width: 100%;
            max-width: 100%;
            height: auto;
            max-height: 300px;
            object-fit: contain; /* Ensure the image fits within the container while maintaining aspect ratio */
            border-radius: 15px;
            margin-bottom: 20px;
        }

        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #3498db;
            color: white;
        }

        /* Prediction Button */
        .prediction-button {
            background-color: #2ecc71;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 15px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }

        /* File Uploader */
        .file-uploader {
            font-weight: bold;
            color: #3498db;
        }

        /* Results Section */
        .results {
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 10px;
        }

        /* Header */
        .header-text {
            font-size: 28px;
            color: #2980b9;
            font-weight: bold;
            text-align: center;
        }

    </style>
""", unsafe_allow_html=True)

# Add banner image with proper fitting from the provided URL
st.markdown("""
    <img src='https://camo.githubusercontent.com/6dd85c8b66f515de1cc4ab0076714d25d27b6f1d58a4b35f142593e670962ad2/68747470733a2f2f6c6f676f6469782e636f6d2f6c6f676f2f3738373236302e706e67'
    class='banner-img' alt='Sentiment Analysis App Banner'/>
""", unsafe_allow_html=True)




# Function for single text prediction
def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

# Function for bulk prediction from CSV
def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions

    graph = get_distribution_graph(data)

    return data, graph

# Function to map prediction output
def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"

# Function to create a pie chart for sentiment distribution
def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    st.pyplot(fig)

# Streamlit app code starts here
def main():
    st.markdown("<h1 class='header-text'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
    st.write("Predict sentiment from text or CSV files.")

    # Sidebar for navigation
    options = st.sidebar.selectbox("Choose Prediction Type", ("Single Text", "Bulk CSV Prediction"))

    if options == "Single Text":
        st.header("Single Sentence Prediction")
        text_input = st.text_area("Enter the text for sentiment analysis:")
        
        if st.button("Predict Sentiment", key="predict_button"):
            if text_input:
                result = single_prediction(predictor, scaler, cv, text_input)
                st.success(f"Predicted Sentiment: {result}")
            else:
                st.error("Please enter a text input.")

    elif options == "Bulk CSV Prediction":
        st.header("Bulk CSV Prediction")
        file = st.file_uploader("Upload a CSV file for bulk sentiment prediction", type=["csv"], help="Upload a CSV with a 'Sentence' column", key="file_uploader")

        if file is not None:
            data = pd.read_csv(file)
            if "Sentence" in data.columns:
                st.subheader("Sentiment Distribution")
                data, graph = bulk_prediction(predictor, scaler, cv, data)
                # Display the sentiment distribution graph

                # Display the predicted sentiments
                st.subheader("Predicted Sentiments")
                st.dataframe(data)

                # Download the results as a CSV
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64 for download
                href = f'<a href="data:file/csv;base64,{b64}" download="Predictions.csv" class="file-uploader">Download Predicted Sentiments CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("CSV file must contain a 'Sentence' column.")


if __name__ == "__main__":
    main()
