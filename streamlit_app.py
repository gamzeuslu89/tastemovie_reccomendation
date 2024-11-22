import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests


# Veri setini yükle
@st.cache_data  # Yeni önbellekleme komutu
def load_data():
    Movie_df = pd.read_csv(
        'Movie_df.csv')  # varsayılan olarak csv dosyasını kullanıyorum, dosya adınızı değiştirebilirsiniz
    return Movie_df


# TF-IDF vektörlerini oluştur
@st.cache_data  # Yeni önbellekleme komutu
def create_tfidf_matrix(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['tags'].values.astype('U'))
    return tfidf_matrix


# Kosinüs benzerliğini hesapla
@st.cache_data  # Yeni önbellekleme komutu
def calculate_cosine_similarity(_tfidf_matrix):
    cosine_sim = cosine_similarity(_tfidf_matrix, _tfidf_matrix)
    return cosine_sim


# Film önerilerini al
@st.cache_data  # Yeni önbellekleme komutu
def get_recommendations(title, data, cosine_sim):
    idx = data[data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    recommended_movies = [data['title'].iloc[i[0]] for i in sim_scores]
    return recommended_movies


# Afişleri al
def fetch_poster(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': 'fd5d08451c6cbb8b3ec7b1815a737821',  # The Movie Database (TMDb) API key
        'query': movie_title
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data['results']:
        movie_id = data['results'][0]['id']
        poster_path = data['results'][0]['poster_path']
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    return None

## Main Page Tasarımı ##

def main():
    st.set_page_config(layout="wide", page_title="TasteMovie", page_icon="🎬")
    st.title(":blue[Taste]Movie")
    st.subheader("TasteMovie: Where Your Movie Taste Comes to Life!")
    st.markdown("""
    <style>
    
    .stButton > button {
        background-color: #FF6347;  /* Daha canlı bir renk tonu */
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    home_tab, recommendation_tab, data_tab = st.tabs(["Home", "Find Something Great", "DataSet&Algorithm"])
    text_col, image_col_left, image_col_right = home_tab.columns([2, 1, 1], gap="small")
    text_col.markdown("""
   
    Welcome! TasteMovie is your gateway to the world of cinema. Whether you're a film enthusiast or someone looking for something new, you'll find recommendations tailored to your taste here.

    🎬 Discover: Don't get lost in the sea of movies! Our unique algorithm and user reviews help you find the perfect film for every mood.

    📚 Get Inspired: From popular ones to hidden ones, everything is just a click away.

    Let’s kickstart your movie journey!
    Get a recommendation, find your next favorite film, and elevate your movie experience.

    With TasteMovie, there’s a movie for everyone!
    
    Click on the "***Find Something Great***" tab above to get some movie recommendations.

    """)
    ## Görselleri ekledim. ##
    image_col_left.image("movie.jpg")
    image_col_right.image("movie2.jpg")


    # Veri setini yükle
    Movie_df = load_data()

    # TF-IDF vektörlerini oluştur
    tfidf_matrix = create_tfidf_matrix(Movie_df)

    # Kosinüs benzerliğini hesapla
    cosine_sim = calculate_cosine_similarity(tfidf_matrix)

## Recommendation Tab ##

    # Film seçme arayüzü
    film_basligi = recommendation_tab.selectbox('**Find Movie Recommendaitons Similar To:**',Movie_df['title'].values)

    if recommendation_tab.button(' Get a Recommendation', icon="🍿"):
        recommendation_tab.markdown(  """
        For more details about movies, visit [IMDB's homepage](https://www.imdb.com).
        """,
        unsafe_allow_html=True)
        # Önerileri al
        recommendations = get_recommendations(film_basligi, Movie_df, cosine_sim)

        # Önerilen filmleri göster
        recommendation_tab.subheader('Recommended Movies')
        cols = recommendation_tab.columns(len(recommendations))  # Kolonları oluştur
        for i, recommendation in enumerate(recommendations):
            with cols[i]:  # Her bir kolona bir film ve afiş ekleyin
                poster_url = fetch_poster(recommendation)
                if poster_url:
                    st.image(poster_url, caption=recommendation, width=200,
                             use_container_width=True)  # Afişleri küçültmek için width parametresini ayarlayın
                else:
                    st.write("No poster found.")

## DataTab ##

    data_tab.title("DataSet & TasteMovie Recommendation Algorithm")
    data_tab.subheader("DataSet Link:")
    data_tab.markdown( """
        For details of the dataset, visit [Here](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data).
        """,)
    data_tab.subheader("Algorithm:")
    data_tab.markdown(""" TasteMovie Recommendation Algorithm

TasteMovie uses an advanced recommendation algorithm to provide users with the most suitable movie suggestions. This algorithm measures the similarities between movies to identify content that may interest the user. 

Here's how the process works below:

**1. TF-IDF (Term Frequency-Inverse Document Frequency) Approach:**

To gather information about movies, we combine their summaries, genres, cast, and other keywords to create unique "tags" for each film. However, some words are more frequently used and less distinctive.
TF-IDF is used to make these tags more meaningful and distinctive:

-TF (Term Frequency): Measures how often a specific word appears in a movie's tag.

-IDF (Inverse Document Frequency): Evaluates how common or rare this word is across all movies.
As a result, common but less informative words have less impact, while rarer and more significant words have a higher weight.

**2.Cosine Similarity:**

Once the TF-IDF vectors are created, Cosine Similarity is applied to measure the similarity between movies.
This method calculates the angle between two movie vectors. A smaller angle indicates higher similarity:

Value Close to 1: The movies are very similar.

Value Close to 0: The movies are very different.

For instance, if you select an action-sci-fi movie, the algorithm will recommend other movies with similar action and sci-fi elements.

**3. User-Focused Recommendation Mechanism**
When a user selects a movie, the algorithm retrieves the TF-IDF vector for that movie and compares it with all other movies. The movies with the highest similarity scores are then ranked and recommended to the user.


**4. Dynamic and Personalized Recommendations**
The algorithm can continuously update to find movies that match users' tastes. For example, as more movies with different genres or themes are tagged, the system can offer a broader range of recommendations.

    """)
if __name__ == "__main__":
    main()