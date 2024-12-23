
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

client_id = '97aeaf1e98f943edb1344ab86f71692a'
client_secret = '9f35e123caa7490b904ad6bcb98f4ba9'
playlistId = '1dtCMTzAOzwKXqklxPJNS'
Y
# 37i9dQZF1DXbrUpGvoi3TS - 1(similar sad songs)
# 1dtCMTYzAOzwKXqklxPJNS - 2(old songs, rock, rap)
# 0IN7IWKmIfwlEysGyWUuRg - 3(mix of modern electronic, pop, and rock)

dataset = []
dataset2 = []
dataset3 = []

def getToken():
    # gabungkan client_id dan client_secret
    auth_string = client_id + ':' + client_secret

    # encode ke base64
    auth_b64 = base64.b64encode(auth_string.encode('utf-8'))

    # url untuk mengambil token
    url = 'https://accounts.spotify.com/api/token'

    # header untuk mengambil token - sesuai dengan guide dari spotify
    headers = {
        'Authorization': 'Basic ' + auth_b64.decode('utf-8'),
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    # data untuk mengambil token - sesuai dengan guide dari spotify
    data = {'grant_type': 'client_credentials'}

    # kirim request POST ke spotify
    result = post(url, headers=headers, data=data)

    # parse response ke json
    json_result = json.loads(result.content)
    token = json_result['access_token']

    # ambil token untuk akses API
    return token

# pengambilan token untuk otorisasi API
def getAuthHeader(token):
    return {'Authorization': 'Bearer ' + token}

# pengambilan audio features dari track (lagu)
def getAudioFeatures(token, trackId):
    url = f'https://api.spotify.com/v1/audio-features/{trackId}'  # endpoint untuk akses playlist
    headers = getAuthHeader(token)  # ambil token untuk otorisasi, gunakan sebagai header
    result = get(url, headers=headers)  # kirim request GET ke spotify
    json_result = json.loads(result.content)  # parse response ke json

    if not json_result:  # Jika respons kosong
        st.error(f"Audio features for track ID {trackId} not found.")
        return

    try:
        # Ambil data yang diperlukan dari response
        audio_features_temp = [
            json_result.get('danceability', 0),  # Default nilai 0 jika key tidak ditemukan
            json_result.get('energy', 0),
            json_result.get('key', 0),
            json_result.get('loudness', 0),
            json_result.get('mode', 0),
            json_result.get('speechiness', 0),
            json_result.get('acousticness', 0),
            json_result.get('instrumentalness', 0),
            json_result.get('liveness', 0),
            json_result.get('valence', 0),
            json_result.get('tempo', 0),
        ]
        dataset2.append(audio_features_temp)
    except KeyError as e:
        st.error(f"Key error: {e} in track ID {trackId}")

# pengambilan track (lagu) dari playlist
def getPlaylistItems(token, playlistId):
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

    # ambil data yang diperlukan dari response
    for i in range(len(json_result['items'])):
        try:
            track = json_result['items'][i]['track']
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
    # muat dataset
    data = pd.read_csv('dataset.csv')
    data
    st.write("### Deleting suffix and prefix from encoding")  # streamlit widget

    # Hapus karakter yang tidak perlu pada kolom artist dan name
    data['artist'] = data['artist'].map(lambda x: str(x)[2:-1])
    data['name'] = data['name'].map(lambda x: str(x)[2:-1])

    st.write("### Deleted empty song names")
    #delete empty string in name column
    data = data[data['name'] != '']

    #reset index
    data = data.reset_index(drop=True)

    st.write("### MinMax Normalization Result")  # streamlit widget
    data2 = data.copy()
    data2 = data2.drop(['artist', 'name', 'year', 'popularity', 'key','duration_ms', 'mode', 'id'], axis=1) 

    x = data2.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data2 = pd.DataFrame(x_scaled)
    

    # convert to dataframe
    data2.columns = ['acousticness','danceability','energy','instrumentalness','loudness', 'liveness', 'speechiness', 'tempo','valence']
    data2   

    st.write("### Dimensionality Reduction with PCA")  # streamlit widget
    pca = PCA(n_components=2)
    pca.fit(data2)
    pca_data = pca.transform(data2)

    # buat dataframe dari hasil pca
    pca_df = pd.DataFrame(data=pca_data, columns=['x', 'y'])

    # plot pca_df using plotly
    fig = px.scatter(pca_df, x='x', y='y', title='PCA')
    st.plotly_chart(fig)  # output plotly chart using streamlit


    st.write("### Clustering with K-Means")  # streamlit widget

    # rubah bentuk data ke list 
    data2 = list(zip(pca_df['x'], pca_df['y']))

    # fit kmeans model
    kmeans = KMeans(n_init=10, max_iter=1000).fit(data2)

    # make scatter plot using plotly
    fig = px.scatter(pca_df, x='x', y='y', color=kmeans.labels_, color_continuous_scale='rainbow', hover_data=[data.artist, data.name])
    st.plotly_chart(fig)  # output plotly chart using streamlit

    st.write("Process Done!")

st.write("Processing playlist...")
st.write(f"Dataset size: {len(dataset)}")

client_id = st.text_input("Enter Client ID")
client_secret = st.text_input("Enter Client Secret")
playlistId = st.text_input("Enter Playlist ID")

# streamlit widgets
if st.button('Create Dataset!'):
    token = getToken()
    getPlaylistItems(token, playlistId)

