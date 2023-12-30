import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize(data, render=False, figsize=(30, 30), weight_scale=10):
    """Draw a graph with improved positioning for isolated nodes."""
    G = nx.Graph()

    connected_techs = set()
    for link in data['links']:
        connected_techs.add(link["source"])
        connected_techs.add(link["target"])
    data['nodes'] = [node for node in data['nodes'] if node["id"] in connected_techs]

    # Add nodes and edges to the graph
    for node in data["nodes"]:
        G.add_node(node["id"])
    for link in data["links"]:
        if "weight" in link:
            G.add_edge(link["source"], link["target"], weight=link["weight"])
        if "value" in link:
            G.add_edge(link["source"], link["target"], weight=link["value"])

    # Identify isolated nodes
    isolated_nodes = list(nx.isolates(G))
    connected_nodes = list(set(G.nodes()) - set(isolated_nodes))

    # Use spring layout for the main graph and then adjust for isolated nodes
    pos = nx.spring_layout(G)

    # If there are isolated nodes, position them in a circle around the graph
    if isolated_nodes:
        G.remove_nodes_from(list(nx.isolates(G)))

    # Draw the graph
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, nodelist=connected_nodes, node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=isolated_nodes, node_size=500, node_color='lightblue', node_shape='s')
    nx.draw_networkx_labels(G, pos, font_size=10)
    edge_widths = [d["weight"] * weight_scale for _, _, d in G.edges(data=True)]
    edge_labels = {(u, v): round(d["weight"], 1) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=0)

    plt.axis("off")
    plt.tight_layout()

    if not render:
        return plt

    plt.show()
