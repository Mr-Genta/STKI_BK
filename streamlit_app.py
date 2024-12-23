import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import base64
from requests import post, get
import json
import csv
from sklearn import preprocessing

# Spotify API credentials (dapat diisi via input Streamlit)
client_id = '46612f3625784726bda85f03f73ddd5e'
client_secret = 'e732dd7852f4498294964036a25be804'
playlistId = '2kLDyojkb4eFvpgN7qppr3'

# Dataset
dataset = []
dataset2 = []
dataset3 = []

def getToken():
    """Fungsi untuk mendapatkan token otorisasi dari Spotify API."""
    auth_string = client_id + ':' + client_secret
    auth_b64 = base64.b64encode(auth_string.encode('utf-8'))
    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + auth_b64.decode('utf-8'),
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {'grant_type': 'client_credentials'}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result.get('access_token')
    if not token:
        st.error("Failed to retrieve token. Check your Client ID and Secret.")
    return token

def getAuthHeader(token):
    """Mengembalikan header otorisasi."""
    return {'Authorization': 'Bearer ' + token}

def getAudioFeatures(token, trackId):
    """Mengambil audio features dari track tertentu."""
    url = f'https://api.spotify.com/v1/audio-features/{trackId}'
    headers = getAuthHeader(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)

    # Validasi ketersediaan kunci audio features
    if not json_result or 'danceability' not in json_result:
        st.warning(f"Audio features for track ID {trackId} not found.")
        return [0] * 11  # Mengembalikan nilai default jika data tidak ditemukan

    try:
        audio_features_temp = [
            json_result.get('danceability', 0),
            json_result.get('energy', 0),
            json_result.get('key', 0),
            json_result.get('loudness', 0),
            json_result.get('mode', 0),
            json_result.get('speechiness', 0),
            json_result.get('acousticness', 0),
            json_result.get('instrumentalness', 0),
            json_result.get('liveness', 0),
            json_result.get('valence', 0),
            json_result.get('tempo', 0)
        ]
        dataset2.append(audio_features_temp)
    except KeyError as e:
        st.error(f"Error while processing audio features: {e}")

def getPlaylistItems(token, playlistId):
    """Mengambil daftar track dalam playlist tertentu."""
    url = f'https://api.spotify.com/v1/playlists/{playlistId}/tracks'
    limit = '&limit=100'
    market = '?market=ID'
    fields = '&fields=items(track(id,name,artists,popularity,duration_ms,album(release_date)))'
    url = url + market + fields + limit
    headers = getAuthHeader(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)

    if 'items' not in json_result:
        st.error("No items found in the playlist. Please check the Playlist ID.")
        return

    for i, item in enumerate(json_result['items']):
        try:
            track = item['track']
            playlist_items_temp = [
                track['id'],
                track['name'].encode('utf-8'),
                track['artists'][0]['name'].encode('utf-8'),
                track['popularity'],
                track['duration_ms'],
                int(track['album']['release_date'][0:4])
            ]
            dataset.append(playlist_items_temp)
        except KeyError as e:
            st.warning(f"Skipping track due to missing data: {e}")

    for i in range(len(dataset)):
        getAudioFeatures(token, dataset[i][0])

    for i in range(len(dataset)):
        dataset3.append(dataset[i] + dataset2[i])

    with open('dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "id", "name", "artist", "popularity", "duration_ms", "year",
            "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo"
        ])
        writer.writerows(dataset3)

    dataProcessing()

def dataProcessing():
    """Memproses dataset untuk normalisasi, reduksi dimensi, dan clustering."""
    data = pd.read_csv('dataset.csv')
    data['artist'] = data['artist'].map(lambda x: str(x)[2:-1])
    data['name'] = data['name'].map(lambda x: str(x)[2:-1])
    data = data[data['name'] != '']  # Hapus lagu tanpa nama
    data = data.reset_index(drop=True)

    st.write("### Normalized Dataset")
    data2 = data.drop(['artist', 'name', 'year', 'popularity', 'key', 'duration_ms', 'mode', 'id'], axis=1)
    x = data2.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data2 = pd.DataFrame(x_scaled, columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 
                                            'liveness', 'speechiness', 'tempo', 'valence'])
    st.dataframe(data2)

    st.write("### Dimensionality Reduction with PCA")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data2)
    pca_df = pd.DataFrame(data=pca_data, columns=['x', 'y'])
    fig = px.scatter(pca_df, x='x', y='y', title='PCA Result')
    st.plotly_chart(fig)

    st.write("### Clustering with K-Means")
    data2 = list(zip(pca_df['x'], pca_df['y']))
    kmeans = KMeans(n_init=10, max_iter=1000).fit(data2)
    fig = px.scatter(pca_df, x='x', y='y', color=kmeans.labels_, color_continuous_scale='rainbow',
                     hover_data=[data['artist'], data['name']])
    st.plotly_chart(fig)

    st.write("Process Done!")

st.write("# Spotify Playlist Clustering")
st.write("### Made by Rifky Ariya Pratama")

client_id = st.text_input("Enter Client ID")
client_secret = st.text_input("Enter Client Secret", type="password")
playlistId = st.text_input("Enter Playlist ID")

if st.button('Create Dataset!'):
    if not client_id or not client_secret or not playlistId:
        st.error("Please provide all required inputs!")
    else:
        token = getToken()
        if token:
            getPlaylistItems(token, playlistId)
