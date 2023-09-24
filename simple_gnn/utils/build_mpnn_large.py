from collections import defaultdict


def build_mpnn_large(texts, tokenizer):
    all_x = []
    all_edge_index = []
    all_batch = []
    batch_id = 0

    for text in texts:
        # Tokenize the text
        encoding = tokenizer(text, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=512)
        tokens = encoding['input_ids'][0][1:-1]  # Ignore first and last token ([CLS] and [SEP])

        # Prepare the edges, ignoring the direction of the edge
        edge_weights = defaultdict(int)
        token_ids_to_new_ids = {token_id: new_id for new_id, token_id in
                                enumerate(tokens.tolist())}  # Map original token ids to new ids
        edge_index = []
        for i in range(len(tokens) - 1):  # Exclude the last token
            # Edge between current token and next
            edge = [token_ids_to_new_ids[tokens[i].item()], token_ids_to_new_ids[tokens[i + 1].item()]]
            edge_index.append(edge)

        # Prepare the node features (x) - in this case, we'll use a one-hot encoding
        x = [(token_id / 100000) for token_id in tokens.tolist()]

        # Prepare the batch tensor - since we have only one graph per text message, all nodes belong to the same graph
        batch = [batch_id] * len(token_ids_to_new_ids)

        all_x.extend(x)
        all_edge_index.extend(edge_index)
        all_batch.extend(batch)

        batch_id += 1

    return all_x, all_edge_index, all_batch
