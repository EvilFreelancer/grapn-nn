import json
import os
from urllib.request import urlopen
import validators
import pyarrow as pa
import pyarrow.parquet as pq

JD_ANALYSIS_PATH = 'https://raw.githubusercontent.com/ZhongTr0n/JD_Analysis/main/'
JD_ANALYSIS_GRAPHS = ["jd_data.json", "jd_data2.json"]


def load_json(path):
    if validators.url(path):
        with urlopen(path) as url:
            return json.load(url)
    elif os.path.exists(path):
        with open(path) as file:
            return json.load(file)
    else:
        raise FileNotFoundError("File does not exist")


def load_jda(name: str):
    if name not in JD_ANALYSIS_GRAPHS:
        raise Exception(f"Provided {name} is not allowed")
    return load_json(JD_ANALYSIS_PATH + name)


def convert_dec_to_bits(number: int, size: int = 8):
    return [int(bit) for bit in bin(number)[2:].zfill(size)]


def get_nodes(graph_data):
    # Get basic information about graph
    node_list = [node['id'] for node in graph_data['nodes']]
    node_index = {index: node for index, node in enumerate(node_list)}
    node_mapping = {node_id: i for i, node_id in enumerate(node_list)}

    # Convert IDs to binary arrays
    node_feat = []
    for i in range(len(node_list)):
        node_feat.append(convert_dec_to_bits(i))

    return node_list, node_index, node_mapping, node_feat


def get_edges(node_mapping, graph_data):
    # Remap edges and extract weights
    edge_index = [[node_mapping[link['source']], node_mapping[link['target']]] for link in graph_data['links']]
    edge_attr = [link['value'] for link in graph_data['links']]

    return edge_index, edge_attr


def build_jda_large_table():
    # Keys in which we're interesting
    keys = ['node_feat', 'node_index', 'edge_index', 'edge_attr', 'node_list']

    # Init object of list
    t_data = {key: [] for key in keys + ['num_nodes', 'num_edges']}
    for jd in JD_ANALYSIS_GRAPHS:
        # Load graph
        graph_data = load_jda(jd)

        # Get information about nodes and edges
        node_list, node_index, node_mapping, node_feat = get_nodes(graph_data)
        edge_index, edge_attr = get_edges(node_mapping, graph_data)

        # Pass values to arrays
        t_data['node_feat'].append(node_feat)
        t_data['node_index'].append(node_index)
        t_data['edge_index'].append(edge_index)
        t_data['edge_attr'].append(edge_attr)
        t_data['y'].append(node_list)
        t_data['num_nodes'].append(len(node_index))
        t_data['num_edges'].append(len(edge_index))

    return pa.table(
        data={
            'node_feat': t_data['node_feat'],
            'edge_index': t_data['edge_index'],
            'edge_attr': t_data['edge_attr'],
            'y': t_data['y'],
            'num_nodes': t_data['num_nodes'],
            'num_edges': t_data['num_edges'],
        },
        schema=pa.schema([
            pa.field(name="node_feat", type=pa.list_(pa.list_(pa.int8())),
                     metadata={"description": "List of nodes features (decimals converted to 8bit array)"}),
            pa.field(name="edge_index", type=pa.list_(pa.list_(pa.int64())),
                     metadata={"description": "List of edges between nodes"}),
            pa.field(name="edge_attr", type=pa.list_(pa.float64()),
                     metadata={"description": "Attributes of nodes"}),
            pa.field(name="y", type=pa.list_(pa.string()),
                     metadata={"description": "List of node labels"}),
            pa.field(name="num_nodes", type=pa.int64(),
                     metadata={"description": "Number of nodes"}),
            pa.field(name="num_edges", type=pa.int64(),
                     metadata={"description": "Number of edges"}),
        ]),
    )


# def build_subgraphs_table(large_graph, num_subgraphs=10):
#     subgraphs = generate_subgraphs(large_graph, num_subgraphs)
#     print(subgraphs)
#     exit()


def save_jd_dataset():
    large_table = build_jda_large_table()
    pq.write_to_dataset(large_table, root_path='data/large')
