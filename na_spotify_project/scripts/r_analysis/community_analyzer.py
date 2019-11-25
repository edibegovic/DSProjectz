# imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import collections, os
from sklearn import metrics
import pandas as pd

# variables

genres = ['house', 'pop', 'hip hop', 'rap', 'classical', 'rock', 'tech', 'indie', 'metal', 'edm', 'step', 'elec', 'jazz']

fast_greedy_file = 'cluster_fast_greedy_groups.csv'
infomap_file = 'cluster_infomap_groups.csv'
label_prop_file = 'cluster_label_prop_groups.csv'
louvain_file = 'cluster_louvain_groups.csv'
leading_eigen_file = 'cluster_leading_eigen_groups.csv'
walktrap_file = 'cluster_walktrap_groups.csv'

def parse_groups(filename):
    """
    Parse the groups file provided from igraph into a dictionary
    """
    groups_dict = dict()
    counter = 0
    with open(filename, 'r') as this_file:
        for line in this_file.readlines():
            if line.strip() == '"x"':
                counter += 1
            elif line.strip() != '':
                groups_dict[line.strip().split(',')[1].replace('"', '')] = counter
    return groups_dict

def check_similarity(community_dict, pruned_graph):
    """
    Check for similarity between iraph and genres
    """
    group_list = []
    genre_list = []
    pop_list = []
    for node in pruned_graph.nodes(data = True):
        if node[0].split(':')[2] in community_dict:
            genre_list.append(node[1]['genres'])
            group_list.append(community_dict[node[0].split(':')[2]])
            pop_list.append(node[1]['popularity'])
   # print(f"group list: {len(group_list)}")
   # print(f"genre list: {len(genre_list)}")
   # print(f"popularity list: {len(pop_list)}")

    group_array = np.array(group_list)
    genre_array = np.array(genre_list)
    pop_array = np.array(pop_list)

    pop_df = pd.DataFrame(data = {'popularity' :pop_array, 'group': group_array, 'genre': genre_array})
    print(pop_df)

    myFig = plt.figure(figsize = (12,6))
    grouped = pop_df['popularity'].groupby(by = 'group')
    grouped.boxplot()
    myFig.savefig("pop_group.png")

    #print(metrics.normalized_mutual_info_score(np.array(group_list), np.array(genre_list)))

    # check for genre distribution
    for group_id in collections.Counter(group_list).keys():
        current_list = []
        for idx,current_id in enumerate(group_list):
            if group_id == current_id:
                current_list.append(genre_list[idx])
       # print(collections.Counter(current_list))


def read_pickle_graph():
    """
    Goes to data directory reads graph, returns it
    """
    pickle_file_path = "/".join(os.path.abspath(os.path.curdir).split('/')[:-2]) + "/data/spotify_data.pickle"

    return nx.read_gpickle(pickle_file_path)

def prune_network(graph_object, min_degree = 1):
    """
    Given graph prunes nodes with less than given degree
    """
    nodes_to_prune = set()
    for node in graph_object.nodes(data = True):
        this_node_genre = ''
        current_genre_dict = dict()
        try:
            if (graph_object.degree(node[0]) <= min_degree) or len(node[1]['genres']) == 0:
                nodes_to_prune.add(node[0])
            else:
                for genre_in_node in node[1]['genres']:
                    for chosen_genre in genres:
                        if chosen_genre in genre_in_node:
                            current_genre_dict[chosen_genre] = current_genre_dict.get(chosen_genre, 0) + 1
                sorted_x = sorted(current_genre_dict.items(), key = lambda kv: kv[1])
                if len(sorted_x) > 0:
                    node[1]['genres'] = sorted_x[::-1][0][0]
                if len(current_genre_dict) == 0:
                    nodes_to_prune.add(node[0])
        except KeyError:
            nodes_to_prune.add(node[0])
    
    for node in nodes_to_prune:
        graph_object.remove_node(node)

    # additionally remove all nodes that have zero degree after this pruning
    zero_deg_nodes = []
    for node in graph_object.nodes():
        if graph_object.degree(node) == 0:
            zero_deg_nodes.append(node)

    for node in zero_deg_nodes:
        graph_object.remove_node(node)

    return graph_object

if __name__ == '__main__':
    plt.style.use("ggplot")
    sp_graph = read_pickle_graph()
    pruned_graph = prune_network(sp_graph)
    print(len(pruned_graph.nodes()))
    this_groups_dict = parse_groups(louvain_file)
    check_similarity(this_groups_dict, pruned_graph)
