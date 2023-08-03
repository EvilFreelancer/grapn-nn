from simple_gnn.utils.build_graph_dataset import build_graph_dataset
from transformers import BertTokenizer

# Initialize Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Build graph dataset
input_csv_file = 'train_data39473.csv'
output_jsonl_file = 'graph_dataset.jsonl'
build_graph_dataset(input_csv_file, output_jsonl_file, tokenizer)
