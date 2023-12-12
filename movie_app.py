import streamlit as st
import pickle
import pandas as pd
import requests

import base64



def fetch_poster(movie_id):
    url="https://api.themoviedb.org/3/movie/{0}?api_key=25d8f6f94e3d7f9ef4b3b9195aa8d1fd&language=en-US".format(movie_id)
    data=requests.get(url)
    data=data.json()
    poster_path=data["poster_path"]
    full_path="https://image.tmdb.org/t/p/w500/"+poster_path
    return full_path

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    l = []
    for i in movies_list:
        # print(i[0])
        l.append(i[0])
    suggested_movie = []
    movie_poster=[]
    for i in l:
        movie_id=movies.iloc[i].id
        movie_poster.append(fetch_poster(movie_id))
        suggested_movie.append(movies.iloc[i].title)
        #print("Recommanded movies are ", movies.iloc[i].title)

    return suggested_movie , movie_poster

movie_dict=pickle.load(open("movie_dict.pkl","rb"))
movies=pd.DataFrame(movie_dict)

movies_data=pickle.load(open("movies_data.pkl","rb"))
movies_data=pd.DataFrame(movies_data)

similarity=pickle.load(open("similarity.pkl","rb"))

st.title("Movie Recommander System")
rad=st.sidebar.radio("Navigator",["Search","Movie Detail"])
user_chosen_movie_name = st.selectbox("Search movie", movies["title"].values)
if rad=="Search":



    if st.button("Recommand"):
        suggested_movies,movie_poster=recommend(user_chosen_movie_name)
        col1,col2,col3,col4,col5=st.columns(5)
        with col1:
            st.text(suggested_movies[0])
            st.image(movie_poster[0])
        with col2:
            st.text(suggested_movies[1])
            st.image(movie_poster[1])
        with col3:
            st.text(suggested_movies[2])
            st.image(movie_poster[2])
        with col4:
            st.text(suggested_movies[3])
            st.image(movie_poster[3])
        with col5:
            st.text(suggested_movies[4])
            st.image(movie_poster[4])


        # for i in range(5):
        #     st.markdown(suggested_movies[i])

if rad=="Movie Detail":
    def get_movie_name():
        movie_name,movie_poster = recommend(user_chosen_movie_name)
        return movie_name


    suggested_movies = get_movie_name()

    if st.button(suggested_movies[0]):
        actors=movies_data[movies_data["title"]==suggested_movies[0]]["cast"]
        director = movies_data[movies_data["title"] == suggested_movies[0]]["crew"]
        st.write("actors",actors)
        st.write("director",director)
    if st.button(suggested_movies[1]):
        actors=movies_data[movies_data["title"]==suggested_movies[1]]["cast"]
        director = movies_data[movies_data["title"] == suggested_movies[1]]["crew"]
        st.write("actors",actors)
        st.write("director",director)
    if st.button(suggested_movies[2]):
        actors=movies_data[movies_data["title"]==suggested_movies[2]]["cast"]
        director = movies_data[movies_data["title"] == suggested_movies[2]]["crew"]
        st.write("actors",actors)
        st.write("director",director)
    if st.button(suggested_movies[3]):
        actors=movies_data[movies_data["title"]==suggested_movies[3]]["cast"]
        director = movies_data[movies_data["title"] == suggested_movies[3]]["crew"]
        st.write("actors",actors)
        st.write("director",director)
    if st.button(suggested_movies[4]):
        actors=movies_data[movies_data["title"]==suggested_movies[4]]["cast"]
        director = movies_data[movies_data["title"] == suggested_movies[4]]["crew"]
        st.write("actors",actors)
        st.write("director",director)
#
#



