library(igraph)

processFiles <- function(edge_filepath, node_filepath, my_graph){

	# first process nodes and attributes
	artist_ids = c()
	artist_names = c()
	genres = c()
	popularities = c()
	wrong_artists = c()

	nodes_con = file(node_filepath, "r")
	while (TRUE){
		line = readLines(nodes_con, n = 1)
		if (length(line) == 0){
			break
		}
		a = strsplit(line, ";")[[1]]
		if (length(a) == 4){
			artist_id <- strsplit(a[1], ':')[[1]][3]
			artist_ids <- c(artist_ids, artist_id)
			artist_names <- c(artist_names, a[2])
			genres <- c(genres, a[3])
			popularities <- c(popularities, a[4])
		} else{
			wrong_artists <- c(wrong_artists, strsplit(a[1], ':')[[1]][3])
		}
	}
	close(nodes_con)
	my_graph <- my_graph + vertices(artist_ids)
	my_graph <- set_vertex_attr(my_graph, 'artistname', V(my_graph), artist_names)
	my_graph <- set_vertex_attr(my_graph, 'genre', V(my_graph), genres)
	my_graph <- set_vertex_attr(my_graph, 'popularity', V(my_graph), popularities)
	
	# handle edges
    edges = c()
    edges_con = file(edge_filepath, "r")
    while (TRUE){
        line = readLines(edges_con, n = 1)
        if (length(line) == 0){
            break
        }
        a = strsplit(strsplit(line, ' ')[[1]][1], ':')[[1]][3]
        b = strsplit(strsplit(line, ' ')[[1]][2], ':')[[1]][3]
		if (!is.element(a, wrong_artists) & !is.element(b, wrong_artists)){
        	edges = c(edges, a, b)
		}
    }
    close(edges_con)
	my_graph <- my_graph + edges(edges)
	my_graph <- as.undirected(my_graph)

    return (my_graph)
}


# clustering function
community_detection_in_graph <- function(method = cluster_fast_greedy, graph = g, filename = 'cluster_fast_greedy_groups.csv'){
	comm_object <- method(graph)
	print(sizes(comm_object))
	print(algorithm(comm_object))
	plot(comm_object, graph, col = membership(comm_object), vertex.label = NA, vertex.size = 3)
	lapply(groups(comm_object), function(x) write.table(data.frame(x), filename, append = T, sep = ','))
}

g <- make_empty_graph()
g <- processFiles("final_edge_data.txt", "final_node_data.txt", g)

# this is where community detection happens

# function to write groups to file
# lapply(groups(x), function(x) write.table(data.frame(x), 'filename.csv', append = T, sep = ','))
