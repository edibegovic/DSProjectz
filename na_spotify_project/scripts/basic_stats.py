import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import collections, powerlaw, os
from scipy.stats import linregress

# variables
genres = ['house', 'pop', 'hip hop', 'rap', 'classical', 'rock', 'tech', 'indie', 'metal', 'edm', 'step', 'elec', 'jazz']

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
    axes1.set_title("Linear Scale")
    fig1.savefig("degree_hist_final.png")

    # some basic stats
    print(f"number of nodes:\tN = {len(graph_object.nodes())}")
    print(f"average degree:\t\tk = {degrees_array.mean()}")
    print(f"number of edges:\tE = {degrees_array.mean() * len(degrees_array) / 2}")
    print(f"max degree:\t\tmax k = {degrees_array.max()}")
    print(f"min degree:\t\tmin k = {degrees_array.min()}")
    print(f"median degree: \t\tmedian k = {np.median(degrees_array)}")

    print(f"average clustering coefficient:\t{nx.average_clustering(graph_object)}")
    print(f"global clustering coefficient:\t{nx.transitivity(graph_object)}")

    # build list for cummulative degree distribution
    element_counter = collections.Counter(degrees_array)
    degrees_count_array = np.zeros(degrees_array.max()+1, dtype = 'float64')
    for key, value in element_counter.items():
        degrees_count_array[key] = value
    array_for_log_log_plot = np.flip(np.cumsum(np.flip(degrees_count_array[1:])) / np.sum(degrees_count_array))

    # build similar powerlaw
    # build and plot degree distribution 
    x = []
    y = []
    ordered_count_dict = collections.OrderedDict(sorted(element_counter.items()))
    for key, value in ordered_count_dict.items():
        x.append(key)
        y.append(value)

    x_array = np.array(x, dtype = "int32")
    y_array = np.array(y, dtype = "int32")
    p_y = y_array / np.sum(y_array)

    fig2, axes2 = plt.subplots()
    axes2.plot(x, p_y, 'ro')
    axes2.set_title('Degree Distribution Linear Scale')
    fig2.savefig('deg_dist_lin_final.png')

    fig2, axes2 = plt.subplots()
    axes2.loglog(x, p_y, 'ro')
    axes2.set_title('Degree Distribution Log-log Scale')
    fig2.savefig('deg_dist_log_final.png')

    # plot cummulative on log-log scale
    fig, axes = plt.subplots()
    axes.loglog(array_for_log_log_plot, lw = 4)
    axes.loglog()
    axes.set_title("Log-log scale")
    fig.savefig("cum_deg_dist_log_final.png")

    # plot fit to linear reg on log-log scale
    x_for_reg = np.arange(len(array_for_log_log_plot))
    x_for_reg[0] = 0.001
    slope, intercept, r_value, p_value, std_error = linregress(np.log10(x_array), np.log10(np.cumsum(np.flip(y_array))))
    print(f"intercept: \t{intercept}")
    print(f"slope: \t\t{slope}")
    fig3, axes3 = plt.subplots()
    axes3.scatter(np.log10(x_array), np.log10(np.cumsum(np.flip(y_array))), label = "deg_dist")
    genre_set = set()
    axes3.plot(np.log10(x_array), intercept + slope * np.log10(x_array), 'r--', label = "linear_regression_fit")
    axes3.legend()
    fig3.savefig("lin_reg_fit.png")
    

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

def test_deg_dist(graph_object):
    """
    Check if powerlaw or ecponential is better fit for our degree distribution
    Copied from exercise 3
    """
    degrees = sorted(dict(graph_object.degree()).values())
    results = powerlaw.Fit(degrees, discrete = True)
    R, p = results.distribution_compare('power_law', 'exponential') #R loglikelihood ratio of the two distributions

    if (R > 0) and (p < .05):
        R, p = results.distribution_compare('power_law', 'lognormal')
        print(R, p)
        if p < .05:
            if R > 0:
                print(f"Powerlaw hypotheses preferred (p = {p}, CDF exponent = {results.power_law.alpha}")
            else:
                print(f"Lognormal hypothesis preferred (p = {p}, mu = {results.lognormal.mu}, sigma = {results.lognormal.sigma})")
        else:
            print("Powerlaw and lognormal are indistinguishable.")
    else:
        print("We cannot rule out an exponential fit. Definitely not a power law.")

def write_graph_files_for_r(graph_object):
    """
    Write nodes and edges files to parse for R
    """
    with open('r_analysis/final_node_data.txt', "w") as this_file:
        for current_node in graph_object.nodes(data = True):
            this_file.write(f"{current_node[0]};{current_node[1]['name']};{current_node[1]['genres']};{current_node[1]['popularity']}\n")
    with open('r_analysis/final_edge_data.txt', 'w') as this_file:
        for current_edge in graph_object.edges():
            this_file.write(f"{current_edge[0]} {current_edge[1]}\n")


if __name__ == '__main__':
    sp_graph = read_pickle_graph()
    pruned_graph = prune_network(sp_graph)
    degree_distribution_plotter(pruned_graph)
    test_deg_dist(pruned_graph)
    write_graph_files_for_r(pruned_graph)
