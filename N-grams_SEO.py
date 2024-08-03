import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import streamlit as st
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')

st.title('N-grams SEO')
url = st.text_input('Enter your URL')
default_exclude_words = ['you', 'am', 'a', 'I', 'and', 'are', 'is', 'what', 'why', 'how']
additional_exclude_words = st.text_input('Enter additional words to exclude (comma-separated)').split(',')

def seo_analysis(url, exclude_words):
    # Save the good and the warnings in lists
    good = []
    bad = []
    # Send a GET request to the website
    response = requests.get(url)
    # Check the response status code
    if response.status_code != 200:
        st.error("Error: Unable to access the website.")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the title and description
    title = soup.find('title').get_text() if soup.find('title') else None
    description = soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else None

    # Check if the title and description exist
    if title:
        good.append("Title Exists! Great!")
    else:
        bad.append("Title does not exist! Add a Title")

    if description:
        good.append("Description Exists! Great!")
    else:
        bad.append("Description does not exist! Add a Meta Description")

    # Grab the Headings
    hs = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    h_tags = []
    for h in soup.find_all(hs):
        good.append(f"{h.name}-->{h.text.strip()}")
        h_tags.append(h.name)

    if 'h1' not in h_tags:
        bad.append("No H1 found!")

    # Extract the images without Alt
    for i in soup.find_all('img', alt=''):
        bad.append(f"No Alt: {i}")

    # Extract keywords
    # Grab the text from the body of html
    bod = soup.find('body').text

    # Extract all the words in the body and lowercase them in a list
    words = [i.lower() for i in word_tokenize(bod)]

    # Combine default and additional exclude words, and filter out exclude words and stopwords
    exclude_words = [word.strip().lower() for word in exclude_words]
    exclude_words.extend(default_exclude_words)
    sw = nltk.corpus.stopwords.words('english')
    new_words = [i for i in words if i not in sw and i.isalpha() and i not in exclude_words]

    # Calculate n-grams and frequencies
    bi_grams = ngrams(new_words, 2)
    freq_bigrams = nltk.FreqDist(bi_grams)
    bi_grams_freq = freq_bigrams.most_common(61)

    # Extract the frequency of the words and get the 10 most common ones
    freq = nltk.FreqDist(new_words)
    keywords = freq.most_common(61)

    # Convert lists to DataFrames
    keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
    bigrams_df = pd.DataFrame(bi_grams_freq, columns=['Bigram', 'Frequency'])



    # Display the results in Streamlit tabs
    tab1, tab2 = st.tabs(['N-gram Keywords', 'Bi-gram Keywords'])
    with tab1:
        st.dataframe(keywords_df.style.set_properties(**{'text-align': 'left'}),
                     width=800, height=1200)
    with tab2:
        st.dataframe(bigrams_df.style.set_properties(**{'text-align': 'left'}),
                     width=800, height=1200)

# Call the function to see the results
if url:
    combined_exclude_words = default_exclude_words + additional_exclude_words
    seo_analysis(url, combined_exclude_words)
