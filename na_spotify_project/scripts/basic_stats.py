import networkx as nx
import numpy as np
import os

# variables

def read_pickle_graph():
    """
    Goes to data directory reads graph, returns it
    """
    pickle_file_path = "/".join(os.path.abspath(os.path.curdir).split('/')[:-1]) + "/data/spotify_data.pickle"

    return nx.read_gpickle(pickle_file_path)

def degree_distribution_plotter(graph_object):
    """
    Provided graph object creates plot of degree distribution
    """
    degrees = []
    for node in graph_object.nodes():
        degrees.append(sp_graph.degree(node))
    print(len(degrees))

if __name__ == '__main__':
    sp_graph = read_pickle_graph()
    degree_distribution_plotter(sp_graph)
