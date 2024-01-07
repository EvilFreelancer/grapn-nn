import torch
import pyarrow as pa
from simple_gnn.graph.generate_subgraphs import *
from simple_gnn.dataset.jd_dataset import *
from datasets import Dataset, DatasetDict

JD_ANALYSIS_GRAPHS = ["jd_data.json", "jd_data2.json"]

items = {}
for jd in JD_ANALYSIS_GRAPHS:
    items[jd] = {'object': load_jda(jd)}

for jd in JD_ANALYSIS_GRAPHS:
    graph_data = items[jd]['object']
    items[jd]['node_list'] = [node['id'] for node in graph_data['nodes']]
    items[jd]['node_index'] = {index: node for index, node in enumerate(items[jd]['node_list'])}
    items[jd]['node_mapping'] = {node_id: i for i, node_id in enumerate(items[jd]['node_list'])}

for jd in JD_ANALYSIS_GRAPHS:
    node_list = items[jd]['node_list']
    items[jd]['node_feat'] = torch.randn(len(node_list), 1)

for jd in JD_ANALYSIS_GRAPHS:
    graph_data = items[jd]['object']
    node_mapping = items[jd]['node_mapping']
    items[jd]['edge_index'] = [[node_mapping[link['source']], node_mapping[link['target']]] for link in graph_data['links']]
    items[jd]['edge_attr'] = [link['value'] for link in graph_data['links']]

# Ключи которые нас интересуют
keys = ['node_feat', 'node_index', 'edge_index', 'edge_attr', 'node_list']

# Инициализируем объект списков
t_data = {key: [] for key in keys + ['num_nodes', 'num_edges']}

# Пройдёмся по каждому подготовленному дотасету
for jd in JD_ANALYSIS_GRAPHS:
    for key in keys:
        if key == 'node_feat':
            t_data[key].append(items[jd][key].tolist())
        else:
            t_data[key].append(items[jd][key])

    # Посчитаем 'num_nodes' и 'num_edges'
    t_data['num_nodes'].append(len(items[jd]['node_index']))
    t_data['num_edges'].append(len(items[jd]['edge_index']))

# Соберём таблицу с двумя графами
large_table = pa.table(
    data={
        'name': JD_ANALYSIS_GRAPHS,
        'node_feat': t_data['node_feat'],
        'edge_index': t_data['edge_index'],
        'edge_attr': t_data['edge_attr'],
        'y': t_data['node_list'],
        'num_nodes': t_data['num_nodes'],
        'num_edges': t_data['num_edges'],
    },
    schema=pa.schema([
        pa.field(name="name", type=pa.string()),
        pa.field(name="node_feat", type=pa.list_(pa.list_(pa.float64()))),
        pa.field(name="edge_index", type=pa.list_(pa.list_(pa.int64()))),
        pa.field(name="edge_attr", type=pa.list_(pa.float64())),
        pa.field(name="y", type=pa.list_(pa.string())),
        pa.field(name="num_nodes", type=pa.int64()),
        pa.field(name="num_edges", type=pa.int64()),
    ]),
)


# Сгенерируем 1000 подграфов из каждого большого графа
for jd in JD_ANALYSIS_GRAPHS:
    keys = ['name', 'node_feat', 'node_index', 'edge_index', 'edge_attr', 'node_list']
    t_data = {key: [] for key in keys + ['num_nodes', 'num_edges']}
    graph_data = items[jd]['object']
    subgraphs = generate_subgraphs(graph_data, 1000, 2, 15)
    for (index, subgraph) in enumerate(subgraphs):

        # Build edge index
        subgraph_edge_index = []
        for link in subgraph['links']:
            source_idx = items[jd]['node_mapping'].get(link['source'])
            target_idx = items[jd]['node_mapping'].get(link['target'])
            # Add edge only if both nodes are on the subgraph
            if source_idx is not None and target_idx is not None:
                subgraph_edge_index.append([source_idx, target_idx])

        # Build edges weights
        subgraph_edge_attr = [link['value'] for link in subgraph['links']]

        # List of nodes
        subgraph_node_list = [link['id'] for link in subgraph['nodes']]

        # Convert subgraphs nodes of the small graph
        subgraph_node_index = []
        for link in subgraph['nodes']:
            node_idx = items[jd]['node_mapping'].get(link['id'])
            if node_idx is not None:
                subgraph_node_index.append(node_idx)

        # Make a mask for the subgraph nodes
        subgraph_mask = torch.zeros_like(items[jd]['node_feat'])
        for idx in subgraph_node_index:
            subgraph_mask[idx] = 1
        subgraph_masked_features = items[jd]['node_feat'] * subgraph_mask

        t_data['name'].append(jd)
        t_data['node_feat'].append(subgraph_masked_features.tolist())
        t_data['node_index'].append(subgraph_node_index)
        t_data['node_list'].append(subgraph_node_list)
        t_data['edge_index'].append(subgraph_edge_index)
        t_data['edge_attr'].append(subgraph_edge_attr)
        t_data['num_nodes'].append(len(subgraph_node_index))
        t_data['num_edges'].append(len(subgraph_edge_index))

        items[jd]['subgraph_table'] = pa.table(
            data={
                'name': t_data['name'],
                'node_feat': t_data['node_feat'],
                'edge_index': t_data['edge_index'],
                'edge_attr': t_data['edge_attr'],
                'y': t_data['node_list'],
                'num_nodes': t_data['num_nodes'],
                'num_edges': t_data['num_edges'],
            },
            schema=pa.schema([
                pa.field(name="name", type=pa.string()),
                pa.field(name="node_feat", type=pa.list_(pa.list_(pa.float64()))),
                pa.field(name="edge_index", type=pa.list_(pa.list_(pa.int64()))),
                pa.field(name="edge_attr", type=pa.list_(pa.float64())),
                pa.field(name="y", type=pa.list_(pa.string())),
                pa.field(name="num_nodes", type=pa.int64()),
                pa.field(name="num_edges", type=pa.int64()),
            ]),
        )

# Преобразование объектов PyArrow в объекты датасета Hugging Face
ds_large = Dataset(large_table)
ds_jd_data = Dataset(items["jd_data.json"]['subgraph_table'])
ds_jd_data2 = Dataset(items["jd_data2.json"]['subgraph_table'])

# Объединение отдельных датасетов в один датасет с разными сплитами
dataset_dict = DatasetDict({
    'train': ds_large,
    'jd_data.json': ds_jd_data,
    'jd_data2.json': ds_jd_data2
})

# Выгрузка датасета на Hugging Face
dataset_dict.push_to_hub('evilfreelancer/jd_analysis_graphs')
