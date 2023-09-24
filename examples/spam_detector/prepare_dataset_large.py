import json
import csv
from transformers import BertTokenizer
from simple_gnn.utils.build_mpnn_large import build_mpnn_large

# Initialize Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Build graph dataset
input_csv_file = 'train_data_43590.csv'
output_jsonl_file = 'mpnn_dataset.jsonl'

# Read all messages and labels
with open(input_csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    data = [(row[2], int(row[3] == 'spam')) for row in reader if len(row[2]) > 0]

# Separate messages and labels
messages, labels = zip(*data)

# Build graph
x, edge_index, batch = build_mpnn_large(messages, tokenizer)

# Write graph to file
graph_data = {
    'x': x,
    'edge_index': edge_index,
    'batch': batch,
    'y': labels,
}
with open(output_jsonl_file, 'w') as outfile:
    json.dump(graph_data, outfile)
