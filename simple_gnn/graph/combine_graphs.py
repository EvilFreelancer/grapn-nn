def combine_graphs(graph1, graph2):
    combined_nodes = []
    combined_links = []

    # Extract nodes from graphs
    for node in graph1["nodes"]:
        combined_nodes.append({"id": node["id"], "weight": node["weight"]})
    for node in graph2["nodes"]:
        found_node = next((n for n in combined_nodes if n["id"] == node["id"]), None)
        if found_node:
            # If node already exists, add weights
            found_node["weight"] += node["weight"]
        else:
            combined_nodes.append({"id": node["id"], "weight": node["weight"]})

    # Extract links from graphs
    for link in graph1["links"]:
        combined_links.append({"source": link["source"], "target": link["target"], "weight": link["weight"]})
    for link in graph2["links"]:
        if {"source": link["source"], "target": link["target"]} not in [{"source": l["source"], "target": l["target"]} for l in combined_links]:
            combined_links.append({"source": link["source"], "target": link["target"], "weight": link["weight"]})

    # Combine links with same source and target
    for link1 in graph1["links"]:
        for link2 in graph2["links"]:
            if link1["source"] == link2["source"] and link1["target"] == link2["target"]:
                combined_links.remove({"source": link1["source"], "target": link1["target"], "weight": link1["weight"]})
                combined_links.append({"source": link1["source"], "target": link1["target"], "weight": link1["weight"] + link2["weight"]})

    return {
        "nodes": combined_nodes,
        "links": combined_links
    }


if __name__ == "__main__":
    graph1 = {
        "nodes": [{"id": "A", "weight": 1}, {"id": "B", "weight": 2}, {"id": "C", "weight": 1}],
        "links": [{"source": "A", "target": "B", "weight": 0.5}, {"source": "B", "target": "C", "weight": 0.8}]
    }
    graph2 = {
        "nodes": [{"id": "B", "weight": 3}, {"id": "C", "weight": 1}],
        "links": [{"source": "B", "target": "C", "weight": 0.7}]
    }
    combined_graph = combine_graphs(graph1, graph2)
    print(combined_graph)
