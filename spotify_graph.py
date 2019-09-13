
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

result = spotify.search(q="artist:" + "Jamie XX", type="artist", limit=1)["artists"]["items"][0]
print(result['uri'])

artists_done = set()
albums_done = set()
artist_queue = [result['uri']]

while len(artists_done) < 3:
    print(len(artists_done))

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
            print('Adding:: ' + album['name'])
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
                        print('\t\t' + artist['name'] + artist['genre'])
                        queue.put(artist['uri'])
                        if artist['uri'] not in G:
                            # Get detailed description of artist and create node
                            artist = spotify.artist(artist['uri'])
                            G.add_node(artist['uri'], name=artist['name'], popularity=artist['popularity'])
                            # Try adding artist's image
                        # Count how many collaborations
                        try:
                            G[artist['uri']][artist_uri]['freq'] += 1
                        except KeyError:
                            G.add_edge(artist['uri'], artist_uri, freq=1









# def get_genres(artist):
#     result = spotify.search(q="artist:" + artist, type="artist", limit=1)["artists"]["items"][0]
#     print(result['name'] + ": " + str(result['genres']))
