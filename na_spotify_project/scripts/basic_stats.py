import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
import collections

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
    degrees_array = np.array(degrees, dtype = "int64")

    #plot degree histogram
    plt.style.use('ggplot')
    fig1, axes1 = plt.subplots()
    axes1.hist(degrees_array, bins = 100)
    fig1.savefig("degree_hist.png")

    # some basic stats
    print(f"average degree:\t\tk = {degrees_array.mean()}")
    print(f"number of edges:\tE = {degrees_array.mean() * len(degrees_array) / 2}")
    print(f"max degree:\t\tmax k = {degrees_array.max()}")
    print(f"min degree:\t\tmin k = {degrees_array.min()}")
    print(f"median degree: \t\tmedian k = {np.median(degrees_array)}")

    # build list for degree distribution
    element_counter = collections.Counter(degrees_array)
    degrees_count_array = np.zeros(degrees_array.max()+1, dtype = 'float64')
    for key, value in element_counter.items():
        degrees_count_array[key] = value
    array_for_log_log_plot = np.flip(np.cumsum(np.flip(degrees_count_array[1:])) / np.sum(degrees_count_array))
    #print(array_for_log_log_plot)

    fig, axes = plt.subplots()
    axes.loglog(array_for_log_log_plot)
    fig.savefig("degree_dist_log_log_scale.png")

def prune_network(graph_object, min_degree = 1):
    """
    Given graph prunes nodes with less than given degree
    """
    nodes_to_prune = []
    for node in graph_object.nodes():
        if (graph_object.degree(node) <= min_degree):
            nodes_to_prune.append(node)

    for node in nodes_to_prune:
        graph_object.remove_node(node)
    return graph_object

if __name__ == '__main__':
    sp_graph = read_pickle_graph()
    pruned_graph = prune_network(sp_graph)
    degree_distribution_plotter(pruned_graph)
