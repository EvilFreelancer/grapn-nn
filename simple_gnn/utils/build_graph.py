from collections import defaultdict


def build_graph(text, tokenizer):
    # Tokenize the text
    encoding = tokenizer(text, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=512)
    tokens = encoding['input_ids'][0][1:-1]  # Ignore first and last token ([CLS] and [SEP])

    # Prepare the edges, ignoring the direction of the edge
    edge_weights = defaultdict(int)
    token_ids_to_new_ids = {token_id: new_id for new_id, token_id in enumerate(set(tokens.tolist()))}  # Map original token ids to new ids
    for i in range(len(tokens) - 1):  # Exclude the last token
        # Edge between current token and next
        edge = (token_ids_to_new_ids[tokens[i].item()], token_ids_to_new_ids[tokens[i + 1].item()])
        edge_weights[edge] += 1

    # Convert the edge weights into a list of edges
    edges = [[edge[0], edge[1], weight] for edge, weight in edge_weights.items()]

    return edges, len(token_ids_to_new_ids)
