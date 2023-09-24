import torch
from transformers import BertTokenizerFast
from torch_geometric.data import Data
from simple_gnn import GCNSummarizer, GraphSAGESummarizer, GATSummarizer, MLPSummarizer
from simple_gnn.utils.build_graph_dataset import build_graph

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model and tokenizer
model = GCNSummarizer(1200, 800).to(DEVICE)
model.load_state_dict(torch.load(model.file_path))
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def predict_spam(text):
    model.eval()

    # Build graph
    edges, num_nodes = build_graph(text, tokenizer)

    # Convert edges to tensor and create edge index
    edge_index = torch.tensor([(edge[0], edge[1]) for edge in edges], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([edge[2] for edge in edges], dtype=torch.float)

    # Create input data
    data = Data(x=torch.ones((num_nodes, model.conv1.in_channels)), edge_index=edge_index, edge_attr=edge_weight).to(DEVICE)
    batch = torch.zeros(num_nodes, dtype=torch.long).to(DEVICE)  # Create batch tensor

    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr, batch)  # Pass batch tensor to model

    return out.item()


text = "Not everybody know about wiki"
# text = "При этом, в gitflic, раннер зарегистрировался и получил Состояние \"Активный\", в поле агент, после очередной циклической регистрации, изменяется имя. После отключения ранера, в состояниях, висит как Активный."
# text = "Обученная модель весит 711кб, если я правильно понял можно добавить ещё признаков и одной моделью определять насколько текст является скамом, относится ли он к рекламе прона, какую эмоцию несёт."
# text = "БЕСПЛАТНО БОНУСКОЕ ПРЕДЛОЖЕНИЕ - См. Ниже * * * * Мы можем предоставить высочайшее качество, практически идентичные, приходи покупай у нас!"

print(text)
print(predict_spam(text))
