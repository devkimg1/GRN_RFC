#graph_tool is only semi-compatible with Windows and is best used in a Linux environment
from graph_tool import centrality, draw, inference, clustering, generation, search, stats, topology
import graph_tool as gt
import networkx as nx
import numpy as np
import pandas as pd
from pytictoc import TicToc
import matplotlib.pyplot as plt
import pyintergraph


# import importance matrix for all genes trained on TFs
# change file name below to match location of importances_tf_allgenes
adj = pd.read_csv('importances_tf_allgenes.csv').to_numpy()
# make the matrix square - for the graph, this means that nodes will be genes in the network and edges will be TFs

max = np.amax(adj)
print(max)

#set a threshold (percent of maximum in adjacency matrix) to determine edges
threshold = 0.1
for i in range(len(adj)):
    for j in range(len(adj)):
        if adj[i,j] < threshold*max:
            adj[i,j] = 0
        else:
            adj[i,j] = 1

print(adj)

# convert to graph-tool graph
g = nx.from_numpy_matrix(adj)
g = pyintergraph.nx2gt(g, labelname="node_label")


#for regular SBM
state = inference.minimize_blockmodel_dl(g, deg_corr=False, B_max=10, B_min=1)
t = TicToc()
t.tic()
change_count = 0
#run multiple times for minimum description length
for i in range(0, 50):
    state_new = inference.minimize_blockmodel_dl(g, deg_corr=False, B_max=10, B_min=1)
    if state_new.entropy() < state.entropy():
        state = state_new
        change_count = change_count + 1
        print("Updated:\t", change_count)
        print("New minimum description length:\t", state.entropy())
    else:
        pass
t.toc('Fit took')

#name output based on threshold
gt.draw.graph_draw(g, vertex_fill_color=state.get_blocks(), output="sbm10percent.svg")
e = state.get_matrix()
plt.matshow(e.todense())
plt.colorbar()
plt.savefig("sbm_all_10percentthreshold")

#for nested SBM
state = inference.minimize_nested_blockmodel_dl(g, deg_corr=False)
t = TicToc()
t.tic()

#run multiple times to for minimum description length
change_count = 0
for i in range(0, 10):
    state_new = inference.minimize_nested_blockmodel_dl(g, deg_corr=False)
    if state_new.entropy() < state.entropy():
        state = state_new
        change_count = change_count + 1
        print("Updated:\t", change_count)
        print("New minimum description length:\t", state.entropy())
    else:
        pass
t.toc('Fit took')

#name output based on threshold
gt.draw.draw_hierarchy(state, output='nestedsbm_all_10percentthreshold.svg')
