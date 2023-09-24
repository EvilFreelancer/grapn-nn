from simple_gnn.utils.build_graph import build_graph
from transformers import BertTokenizer
import json, csv

# Initialize Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Build graph dataset
input_csv_file = 'train_data_43590.csv'
output_jsonl_file = 'graph_dataset.jsonl'

# Build graph dataset
with open(input_csv_file, 'r') as file, open(output_jsonl_file, 'w') as outfile:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        _, _, message, label, _ = row

        # Skip empty messages
        if len(message) < 1:
            continue

        # Build graph
        graph, num_nodes = build_graph(message, tokenizer)

        # Skip empty graphs
        if len(graph) < 1:
            continue

        # Write graph to file
        graph_data = {
            'graph': graph,
            'label': int(label == 'spam'),
            'num_nodes': num_nodes,
        }
        json.dump(graph_data, outfile)
        outfile.write('\n')
