from ts2vg import NaturalVG
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import ArrowStyle, FancyArrowPatch

def get_named_colors(num_colors):
    """
    Gets a list of named colors from matplotlib, choosing the number of colors
    based on the number of communities.
    """
    color_names = list(mcolors.CSS4_COLORS.keys())
    return [mcolors.to_hex(color_names[i % len(color_names)])
            for i in range(num_colors)]


def plot_weighted_nvg(
    vg,
    ax,
    cbar_ax,
    weights_cmap="coolwarm_r",
    weights_range=(-3.5, 3.5),
):
    bars = ax.bar(vg.xs, vg.ts, color="#ccc", edgecolor="#000", width=0.3)
    ax.set_xticks(vg.xs)

    color_mappable = ScalarMappable(norm=Normalize(*weights_range), cmap=weights_cmap)
    cbar_ax.get_figure().colorbar(color_mappable, cax=cbar_ax, orientation="vertical", aspect=30, pad=0.05)

    for (n1, n2, w) in vg.edges:
        x1, y1 = vg.xs[n1], vg.ts[n1]
        x2, y2 = vg.xs[n2], vg.ts[n2]

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=ArrowStyle("-"),
            shrinkA=0,
            shrinkB=0,
            color=color_mappable.to_rgba(w, alpha=1),
            linewidth=2,
        )

        ax.add_patch(arrow)


def make_visibility_graph(intensities,filepath=None,penetrable_limit=0):
    # 1. Build visibility graph
    """
    Make a visibility graph for a given time series.

    Parameters
    ----------
    intensities : array_like
        The intensity values of the signal.
    filepath : str, optional
        If given, saves the plots to a file at the specified filepath.
    Returns
    -------
    nxg : networkx.Graph
        The visibility graph represented as a networkx graph.
    ks : array_like
        The degree distribution, where ks[i] is the number of nodes of degree i.
    ps : array_like
        The degree distribution, where ps[i] is the fraction of nodes of degree i.
    node_positions : array_like
        The positions of the nodes in the visibility graph, where node_positions[i] is the position of node i.
    penetrable_limit=0 : float, optional
        The penetrable limit of the visibility graph.
    Notes
    -----
    The positions of the nodes in the visibility graph are determined by the Natural Visibility Graph algorithm.
    The degree distribution is calculated as the fraction of nodes of each degree.
    """
    g = NaturalVG(directed=None,penetrable_limit=penetrable_limit).build(intensities)
    nxg = g.as_networkx()

    # 2. Make plots
    fig, [ax0, ax1] = plt.subplots(ncols=1,nrows=2, figsize=(12, 6),height_ratios=(1,5))
    ax0.plot(intensities,linewidth=0.1)
    ax0.set_title("Time Series")

    graph_plot_options = {
        "with_labels": False,
        "node_size": 2,
        "linewdiths":0.01,
        "node_color": [(0, 0, 0, 1)],
        "edge_color": [(0, 0, 0, 0.15)],
    }
    nx.draw_networkx(nxg, ax=ax1, pos=g.node_positions(), **graph_plot_options)
    ax1.tick_params(bottom=True, labelbottom=True)
    ax1.plot(intensities,linewidth=0.01)
    ax1.set_title("Visibility Graph")
    if filepath is not None:
        plt.savefig(filepath+'/Visibility_Graph_time_series.png')
        plt.close()
    else :
        plt.show()

    # 3. Get degree distribution
    ks, ps = g.degree_distribution
    # 4. Make plots
    fig, [ax0, ax1, ax2] = plt.subplots(nrows=3, figsize=(12, 12))

    ax0.plot(intensities, c="#000", linewidth=.1)
    ax0.set_title("Time Series")
    ax0.set_xlabel("t")

    ax1.scatter(ks, ps, s=2, c="#000", alpha=1)
    ax1.set_title("Degree Distribution")
    ax1.set_xlabel("k")
    ax1.set_ylabel("P(k)")

    ax2.scatter(ks, ps, s=2, c="#000", alpha=1)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_title("Degree Distribution (log-log)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("P(k)")
    if filepath is not None:
        plt.savefig(filepath+'/Visibility_Graph_time_series_Degree_Distribution.png')
        plt.close()
    else :
        plt.show()
    return nxg,ks,ps,g.node_positions()

def make_communities(nxg,intensities,locs,range_x=5000,resolution= 0.7,filepath=None,method='louvain'):
    from cdlib import algorithms
    import matplotlib.colors as mcolors

    if method == 'networkx':
        metrics = {}
        louvain_coms = nx.algorithms.community.greedy_modularity_communities(nxg)
    elif method == 'leiden':
        louvain_coms = algorithms.leiden(nxg)
        metrics = {}
        print("Average internal degree of the communities:", louvain_coms.average_internal_degree())
        metrics["average_internal_degree"] = louvain_coms.average_internal_degree()
        print("Erdos Renyi modularity of the communities:", louvain_coms.erdos_renyi_modularity())
        metrics["erdos_renyi_modularity"] = louvain_coms.erdos_renyi_modularity()
        print("Modularity density of the communities:", louvain_coms.modularity_density())
        metrics["modularity_density"] = louvain_coms.modularity_density()
        print("Z-modularity of the communities:", louvain_coms.z_modularity())
        metrics["z_modularity"] = louvain_coms.z_modularity()
        print("Number of communities:", len(louvain_coms.communities))
        metrics["number_of_communities"] = len(louvain_coms.communities)
    elif method == 'louvain':
        louvain_coms = algorithms.louvain(nxg,resolution= resolution,randomize = False)
        metrics = {}
        print("Average internal degree of the communities:", louvain_coms.average_internal_degree())
        metrics["average_internal_degree"] = louvain_coms.average_internal_degree()
        print("Erdos Renyi modularity of the communities:", louvain_coms.erdos_renyi_modularity())
        metrics["erdos_renyi_modularity"] = louvain_coms.erdos_renyi_modularity()
        print("Modularity density of the communities:", louvain_coms.modularity_density())
        metrics["modularity_density"] = louvain_coms.modularity_density()
        print("Z-modularity of the communities:", louvain_coms.z_modularity())
        metrics["z_modularity"] = louvain_coms.z_modularity()
        print("Number of communities:", len(louvain_coms.communities))
        metrics["number_of_communities"] = len(louvain_coms.communities)
    else :
        raise ValueError("method must be 'networkx', 'leiden' or 'louvain'")

    COLORS = get_named_colors(len(louvain_coms.communities))
    node_colors = ["#000000"] * len(intensities)
    for community_id, community_nodes in enumerate(louvain_coms.communities):
        for node in community_nodes:
            node_colors[node] = COLORS[community_id % len(COLORS)]

    fig, [ax0, ax1] = plt.subplots(nrows=2, figsize=(14, 8))
    ax0.plot(intensities,linewidth=0.1)
    ax0.set_title("Time Series")
    plt.xlim([0,1000])
    graph_plot_options = {
        "with_labels": False,
        "node_size": 6,
        "node_color": [node_colors[n] for n in nxg.nodes],
    }

    nx.draw_networkx(nxg, ax=ax1, pos=g.node_positions(), edge_color=[(0, 0, 0, 0.05)], **graph_plot_options)
    ax1.tick_params(bottom=True, labelbottom=True)
    ax1.plot(intensities, linewidth=0.1, c=(0, 0, 0, 0.15))
    ax1.set_title("Visibility Graph")
    plt.xlim([0,range_x])
    if filepath is not None:
        plt.savefig(filepath+'/Visibility_Graph_time_series_Community_Detection_Louvain.png')
        plt.close()
    else :
        plt.show()

    plt.figure(figsize=(12,8))
    plt.scatter(locs[:,1],locs[:,0],c=intensities,s=30,cmap='Grays',
                vmin=intensities.min(), vmax=intensities.max(),alpha=1)
    plt.scatter(locs[:,1],locs[:,0],c=node_colors,s=30,cmap='RdYlBu_r',
            vmin=intensities.min(), vmax=intensities.max(),alpha=0.1)
    plt.title('Hilbert Reconstruction with Hilbert Curve')
    if filepath is not None:
        plt.savefig(filepath+'/Visibility_Graph_time_series_Community_Detection_Louvain_Overplotted_Image.png')
        plt.close()
    else :
        plt.show()
    return louvain_coms,metrics
