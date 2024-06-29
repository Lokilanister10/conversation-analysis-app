# Import necessary libraries
import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import json
import os

# Set environment variables to control the number of threads used by OpenMP
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Set NLTK data path
nltk_data_path = os.path.expanduser('~/.nltk_data')
nltk.data.path.append(nltk_data_path)

# Ensure stopwords are loaded
stop_words = set(stopwords.words('english'))

# Preprocess the text
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
        tokens = word_tokenize(text)  # Tokenize
        tokens = [t for t in tokens if t not in stop_words]  # Remove stopwords
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize
        return ' '.join(tokens)
    else:
        return ""

# Extract conversation texts
def extract_conversation_text(conversation):
    if isinstance(conversation, list):
        return ' '.join([msg['value'] for msg in conversation if isinstance(msg, dict) and 'value' in msg])
    return ""

# Assign topics based on clusters
def assign_topic(cluster):
    topic_dict = {
        0: 'Programming Practices',
        1: 'Regulations and Guidelines',
        2: 'Health and Safety',
        3: 'Business and Management',
        4: 'Technology and Innovations',
        5: 'Education and Learning',
        6: 'Product Information',
        7: 'Misc',
        8: 'Customer Support',
        9: 'Travel and Tourism'
    }
    return topic_dict.get(cluster, 'Misc')

# Keyword-based Sentiment Analysis
positive_keywords = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'positive', 'love', 'like']
negative_keywords = ['bad', 'terrible', 'horrible', 'poor', 'negative', 'hate', 'dislike']
neutral_keywords = ['okay', 'fine', 'neutral', 'average']

def keyword_based_sentiment(text):
    words = word_tokenize(text)
    if any(word in words for word in positive_keywords):
        return 'positive'
    elif any(word in words for word in negative_keywords):
        return 'negative'
    else:
        return 'neutral'

# Streamlit Web App
def main():
    st.title("Conversation Analysis")

    uploaded_file = st.file_uploader("Upload a JSON Lines (.jsonl) file", type="jsonl")
    if uploaded_file is not None:
        try:
            # Read the JSON Lines file
            data = []
            for line in uploaded_file:
                data.append(json.loads(line.decode('utf-8')))
            df = pd.DataFrame(data)

            if st.button('Start Analysis'):
                with st.spinner('Processing...'):
                    # Extract and preprocess conversation texts
                    df['conversation_text'] = df['conversations'].apply(extract_conversation_text)
                    df['preprocessed'] = df['conversation_text'].apply(preprocess_text)
                    df = df[df['preprocessed'].str.strip() != ""]

                    # Vectorize the preprocessed text
                    vectorizer = TfidfVectorizer(max_features=1000)
                    X = vectorizer.fit_transform(df['preprocessed'])

                    # Clustering with K-means
                    num_clusters = 10  # Number of clusters
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    df['cluster'] = kmeans.fit_predict(X)
                    df['topic'] = df['cluster'].apply(assign_topic)

                    # Sentiment Analysis
                    df['user_sentiment'] = df['conversation_text'].apply(keyword_based_sentiment)

                # Screen 1: Counts
                st.header("Counts")
                st.subheader("Topic Counts")
                topic_counts = df['topic'].value_counts().reset_index()
                topic_counts.columns = ['Topic', 'Count']
                st.table(topic_counts)

                st.subheader("Sentiment Counts")
                sentiment_counts = df['user_sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                st.table(sentiment_counts)

                # Screen 2: Sessions
                st.header("Sessions")
                paginated_data = df[['conversation_text', 'topic', 'user_sentiment']]
                paginated_data.reset_index(inplace=True)
                paginated_data.rename(columns={'index': 'Conversation No', 'topic': 'Topic', 'user_sentiment': 'Sentiment'}, inplace=True)

                page_size = 50
                total_pages = (len(paginated_data) // page_size) + 1
                current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)

                start_idx = (current_page - 1) * page_size
                end_idx = start_idx + page_size
                st.table(paginated_data.iloc[start_idx:end_idx])
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a JSON Lines (.jsonl) file to start analysis.")

if __name__ == "__main__":
    main()
