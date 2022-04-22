import networkx as nx
import numpy as np
from tqdm import tqdm


def create_graph(x, skip_offset=2):
    g = nx.Graph()
    D, H, W = x.shape
    MAX = x.max()
    for d, h, w in tqdm(zip(*np.nonzero(x))):
        if d % skip_offset == 0 or \
           h % skip_offset == 0 or \
           w % skip_offset == 0:
            continue
        g.add_node((d, h, w), weight=MAX - x[d, h, w])

        if w + 1 < W:
            if x[d, h, w + skip_offset] != 0.0:
                g.add_edge((d, h, w), (d, h, w + skip_offset))
            if h + skip_offset < H:
                if x[d, h + skip_offset, w + skip_offset] != 0.0:
                    g.add_edge(
                        (d, h, w), (d, h + skip_offset, w + skip_offset)
                    )
                if d + skip_offset < D and \
                   x[d + skip_offset, h + skip_offset, w + skip_offset] != 0.0:
                    g.add_edge(
                        (d, h, w),
                        (d + skip_offset, h + skip_offset, w + skip_offset)
                    )

        if h + skip_offset < H:
            if x[d, h + skip_offset, w] != 0.0:
                g.add_edge((d, h, w), (d, h + skip_offset, w))
            if d + skip_offset < D and \
               x[d + skip_offset, h + skip_offset, w] != 0.0:
                g.add_edge((d, h, w), (d + skip_offset, h + skip_offset, w))

        if d + skip_offset < D:
            if x[d + skip_offset, h, w] != 0.0:
                g.add_edge((d, h, w), (d + skip_offset, h, w))
            if w + skip_offset < w and \
               x[d + skip_offset, h, w + skip_offset] != 0.0:
                g.add_edge((d, h, w), (d + skip_offset, h, w + skip_offset))

    dg = nx.DiGraph(g)
    for u, v in tqdm(dg.edges()):
        dg[u][v]['weight'] = 1. * dg.nodes[v]['weight']
        dg[v][u]['weight'] = 1. * dg.nodes[u]['weight']

    return dg


def sample_points(dg, u, v):
    sampled_points = np.array(nx.dijkstra_path(dg, u, v))
    return sampled_points
