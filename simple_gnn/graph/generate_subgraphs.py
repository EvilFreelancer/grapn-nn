import random


def generate_subgraphs(dataset, num_subgraphs=5, min_nodes=2, max_nodes=5):
    subgraphs = []
    for _ in range(num_subgraphs):
        selected_nodes = []
        while len(selected_nodes) < random.randint(min_nodes, max_nodes):
            if selected_nodes:
                new_node = random.choice(
                    [link['target'] for link in dataset['links'] if link['source'] in {node['id'] for node in selected_nodes}] +
                    [link['source'] for link in dataset['links'] if link['target'] in {node['id'] for node in selected_nodes}]
                )
            else:
                new_node = random.choice(dataset['nodes'])['id']
            if new_node not in {node['id'] for node in selected_nodes}:
                selected_nodes.append({'id': new_node})
        selected_node_ids = {node['id'] for node in selected_nodes}
        selected_links = [link for link in dataset['links'] if link['source'] in selected_node_ids and link['target'] in selected_node_ids]
        subgraphs.append({'nodes': selected_nodes, 'links': selected_links})
    return subgraphs


if __name__ == "__main__":
    graph = {
        "nodes": [
            {"id": "vb.net"},
            {"id": "vb"},
            {"id": "net"},
            {"id": "assembly"},
            {"id": "mvc"},
            {"id": "python"},
            {"id": "linq"},
            {"id": "html"},
            {"id": "sql"}
        ],
        "links": [
            {"source": "assembly", "target": "html"},
            {"source": "assembly", "target": "net"},
            {"source": "assembly", "target": "python"},
            {"source": "assembly", "target": "sql"},
            {"source": "html", "target": "net"},
            {"source": "html", "target": "python"},
            {"source": "html", "target": "sql"},
            {"source": "linq", "target": "mvc"},
            {"source": "linq", "target": "net"},
            {"source": "mvc", "target": "net"},
            {"source": "mvc", "target": "sql"},
            {"source": "net", "target": "python"},
            {"source": "net", "target": "sql"},
            {"source": "net", "target": "vb"},
            {"source": "net", "target": "vb.net"},
            {"source": "vb", "target": "vb.net"}
        ]
    }
    combined_graph = generate_subgraphs(graph)
    import json

    print(json.dumps(combined_graph))
