library(igraph)

g <- make_empty_graph()

processFile <- function(filepath){
    # read file and process it to a vector of edges
    edges = c()
    con = file(filepath, "r")
    while (TRUE){
        line = readLines(con, n = 1)
        if (length(line) == 0){
            break
        }
        a = strsplit(strsplit(line, ' ')[[1]][1], ':')[[1]][3]
        b = strsplit(strsplit(line, ' ')[[1]][2], ':')[[1]][3]
        edges = c(edges, a, b)
    }
    close(con)
    return (edges)
}

build_graph <- function(my_edges){
    # given edgevector returns graph
    my_vertices <- unique(my_edges)
    g <- g + vertices(my_vertices)
    g <- g + edges(my_edges)
    return (g)
}

this_edges <- processFile("bfs.edgelist")
my_graph <- build_graph(this_edges)

print(centr_betw(my_graph, directed = FALSE))
#print(V(g)))
