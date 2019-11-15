
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

G = nx.read_gpickle('spotify_data.pickle')

genres_ = ['step', 'classical', 'elec', 'rock', 'edm', 'tech', 'indie', 'house', 'rap', 'hip hop', 'pop']
temp_set = set()
remove_nodes = set()
for node in G.nodes(data = True):
    try:
        if len(node[1]['genres']) > 0 and G.degree(node[0]) > 1:
            if get_genre(node[1]['genres']) is not None:
                temp_set.add(get_genre(node[1]['genres']))
        else:
            remove_nodes.add(node[0])
        
        if len(temp_set) == 0:
            remove_nodes.add(node[0])
                
        node[1]['genres'] = list(temp_set)
        temp_set = set()
    except:
        remove_nodes.add(node[0])


def get_genre(l):
    for g in genres_:
        if g in ' '.join(l):
            return g
    return None

for r in remove_nodes:
    G.remove_node(r)

len(G)


with open('norm_node2.txt', 'a') as f:
    for node in G.nodes(data = True):
        cont = node[0] + ";" + node[1]['name'] + ";" + node[1]['genres'][0] +  ";" + str(node[1]['popularity']) + "\n"
        f.write(cont)

nx.write_edgelist(G, "norm_edge2.txt")
# nx.write_gpickle(G, "pri_genre.pickle")


for node in G.nodes(data = True):
    try:
        print(f'{node[1]["name"]} has {node[1]["genres"]}')
    except:
        None

# genre_nodes = [[] for g in genres_]
genre_nodes_uri = {g: [n[0] for n in G.nodes(data=True) if n[1]['genres'][0] == g] for g in genres_}

for key, value in genre_nodes_uri.items():
    print(key, len(value))


nodes_to_keep = set()
all_nodes = {n for n in G.nodes}

for g, n in genre_nodes_uri.items():
    if len(n) >= 100:
        nodes_to_keep.update(np.random.choice(n, 100, replace=False))
    else:
        nodes_to_keep.update(n)

nodes_to_remove = all_nodes - nodes_to_keep
G.remove_nodes_from(nodes_to_remove)
plt.hist()

# desgree dist by genre
for key, value in genre_nodes_uri.items():
    # print(value)
    print(key ,(sum([a[1] for a in G.degree(value)])/len(value)))
    # print(key, len(value))

genre_deg_dist =  [8.01, 5.60, 9.84, 4.00, 10.33, 6.80, 3.66, 6.60, 16.00, 7.20, 5.10]
cols = ['#713535', '#D43A3A', '#D0A1A1', '#000000', '#454545', '#B91A8A', '#FF6293', '#008F1D', '#0DC3CA', '#00F2B6', '#008572', '#2E72E2']

plt_data = list(zip(genres_, genre_deg_dist, cols))
plt_data.sort(key=lambda x: x[1])
plt_data = plt_data[::-1]

plt.bar([_[0] for _ in plt_data], [_[1] for _ in plt_data], align='center', alpha=1.0, color=[_[2] for _ in plt_data])
plt.title('Avgrege degree')
plt.show()

# Share of neighbourg nodes (deg 1, 2 etc.) who share the same genre as the top-k nodes
# H = nx.read_gpickle('Network_centrality.pickle')

# for node in H.nodes(data = True):
#     try:
#         print(node[1])
#     except:
#         None

# plt_data = list(zip(genres_, genre_deg_dist, cols))
# plt_data.sort(key=lambda x: x[1])
# plt_data = plt_data[::-1]

# test = [n[1] for n in H.nodes(data=True)]

# test[2]


# ----------------------------------------------------------------------
# Share of neighbourg nodes (deg 1, 2 etc.) who share the same genre as the top-k nodes


genre_nodes_uri = {g: [(n[0], G.degree(n[0])) for n in G.nodes(data=True) if n[1]['genres'][0] == g] for g in genres_}

for g, n in genre_nodes_uri.items():
    print(g)
    n.sort(key=lambda x: x[1])
    n = n[::-1][0:5]
    print(n)

def get_share_of_adj(n):
    node_genres = nx.get_node_attributes(G,'genres')
    curr_genre = node_genres[n][0]
    return len([node_genres[a][0] for a in G[n] if node_genres[a][0] == curr_genre])/len(G[n])


node_genres["spotify:artist:5y2Xq6xcjJb2jVM54GHK3t"][0]
G["spotify:artist:5y2Xq6xcjJb2jVM54GHK3t"]
