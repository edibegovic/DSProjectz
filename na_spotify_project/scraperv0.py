
import networkx as nx
import re
import spotipy
import time
import sys
import spotipy.oauth2 as oauth2
import spotipy

c_id = '26f8a48dc0234bb89171b8b0316cfb37'
cs = 'c9c42ef8492d401497c03c3f4faf819f'
credentials = oauth2.SpotifyClientCredentials(client_id = c_id, client_secret = cs)
token = credentials.get_access_token()
spotify = spotipy.Spotify(auth=token)

G = nx.Graph()

result = spotify.search(q="artist:" + "Avicii", type="artist", limit=1)["artists"]["items"][0]
print(result['uri'])

artists_done = set()
albums_done = set()
artist_queue = [result['uri']]

cnter = 0
while len(artists_done) < 20:
    cnter += 1
    # update token
    if cnter%500 == 0:
        credentials = oauth2.SpotifyClientCredentials(client_id = c_id, client_secret = cs)
        token = credentials.get_access_token()
        spotify = spotipy.Spotify(auth=token)

    print(str(len(artists_done)))
    artist_uri = artist_queue.pop(0)
    if artist_uri in artists_done:
        continue
    artists_done.add(artist_uri)

    # get albums
    results = spotify.artist_albums(artist_uri, album_type='album,single')
    albums = results['items']


    while results['next']:
        results = spotify.next(results)
        albums.extend(results['items'])

    real_albums = dict()
    for album in albums:
        name = re.sub(r'\([^)]*\)|\[[^)]*\]', '', album['name'])
        name = re.sub(r'\W','', name).lower().strip()
        if name not in real_albums:
            real_albums[name] = album

    # Analyze the albums of this artist
    for album in real_albums:
        if album not in albums_done:
            albums_done.add(album)

            results = spotify.album_tracks(real_albums[album]['id'])
            tracks = results['items']
            while results['next']:
                results = spotify.next(results)
                tracks.extend(results['items'])

            # Get collaborating artists in each track
            for track in tracks:
                for artist in track['artists']:
                    if artist['uri'] != artist_uri:
                        artist_queue.append(artist['uri'])
                        if artist['uri'] not in G:
                            print(artist['name'])
                            artist = spotify.artist(artist['uri'])
                            G.add_node(artist['uri'], name=artist['name'], popularity=artist['popularity'], genres=artist['genres'], followers=artist['followers']['total'])
                        G.add_edge(artist['uri'], artist_uri, freq=1)


# G.remove_nodes_from([node for node,degree in dict(G.degree()).items() if degree < 3])
# G.remove_node('spotify:artist:7A0awCXkE1FtSU8B0qwOJQ')

# for node, attr in G.nodes(data=True):
#     print(node, attr)

nx.write_gpickle(G, 'spotify_data.pickle')
# nx.write_edgelist(G,'tessst.txt', data=True)
