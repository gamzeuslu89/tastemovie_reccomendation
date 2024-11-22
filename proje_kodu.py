import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

credits = pd.read_csv("datasets/tmdb_5000_credits.csv")
movies = pd.read_csv("datasets/tmdb_5000_movies.csv")

credits.head()
credits.shape
movies.head()
movies.shape


movies = movies.merge(credits,on='title')
movies.shape

movies.head()
movies.shape

movies.info()

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.isnull().sum()

movies.dropna(inplace=True)

movies.isnull().sum()

movies.iloc[0]['genres']


import ast
def convert(text):
    h = []
    for i in ast.literal_eval(text):
        h.append(i['name'])
    return h

movies['genres'] = movies['genres'].apply(convert)

movies.head()

movies.iloc[0]['keywords']

movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

movies.iloc[0]['cast']


def convert_cast(text):
    c = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            c.append(i['name'])
        counter+=1
    return c


movies['cast'] = movies['cast'].apply(convert_cast)
movies.head()


def pull_director(text):
    D = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            D.append(i['name'])
            break
    return D

movies['crew'] = movies['crew'].apply(pull_director)
movies.head()

movies.iloc[0]['overview']

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head()

movies.iloc[0]['overview']

def remove_space(L):
    S = []
    for i in L:
        S.append(i.replace(" ",""))
    return S


movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

movies.iloc[0]['tags']

Movie_df = movies[['movie_id','title','tags']]

Movie_df['tags'] = Movie_df['tags'].apply(lambda x: " ".join(x))

Movie_df.head()

Movie_df.iloc[0]['tags']

Movie_df['tags'] = Movie_df['tags'].apply(lambda x:x.lower())

print(Movie_df.head)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# TF-IDF vektörlerini oluşturun
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(Movie_df['tags'].values.astype('U'))  # 'tags' sütununu kullanarak vektörleri oluşturuyoruz



# Kosinüs benzerliğini hesaplayın
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    # Film başlığına göre indeksi alın
    idx = Movie_df[Movie_df['title'] == title].index[0]

    # Tüm filmlerin benzerlik skorlarını alın
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Benzerlik skorlarına göre filmleri sıralayın
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # En iyi 10 filmi alın
    sim_scores = sim_scores[1:11]

    # Film indekslerini alın
    movie_indices = [i[0] for i in sim_scores]

    # İlgili film başlıklarını döndürün
    return Movie_df['title'].iloc[movie_indices]


# Önerileri alın
film_basligi = "Total Recall"  # Kullanıcı tarafından seçilen bir film başlığı
print("Film Önerileri:")
print(get_recommendations(film_basligi))