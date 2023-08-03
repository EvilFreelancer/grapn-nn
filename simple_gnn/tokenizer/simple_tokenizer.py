import os
import json
import re
from typing import List, Dict, Any
from collections import Counter


class SimpleTokenizer:
    token2id: Dict[str, int]
    id2token: Dict[int, str]

    def __init__(self, path: str = "./data"):
        self.token_pattern = re.compile(r'\b\w+\b')
        self.path = path

        # Check if the directory exists, if not, create it
        if not os.path.exists(path):
            os.makedirs(path)

        # Check if token maps already exist, if so load them
        if os.path.isfile(f"{path}/token2id.json") and os.path.isfile(f"{path}/id2token.json"):
            self.token2id, self.id2token = self.load()

    def encode(self, text: str) -> List[int]:
        """Tokenize a text and return a list of token ids"""
        # Check if the token2id map exists, if not, raise an error
        if not hasattr(self, 'token2id'):
            raise RuntimeError("Token maps not created or loaded. Use 'create_token_maps' method before tokenizing.")
        words = self.token_pattern.findall(text.lower())
        return [self.token2id[word] for word in words if word in self.token2id]

    def decode(self, tokens: List[int]) -> str:
        """Detokenize a list of token ids and return a string"""
        # Check if the id2token map exists, if not, raise an error
        if not hasattr(self, 'id2token'):
            raise RuntimeError("Token maps not created or loaded. Use 'create_token_maps' method before detokenizing.")
        return " ".join(self.id2token[token] for token in tokens if token in self.id2token)

    def create_token_maps(self, texts: List[str]):
        """Create token maps from a list of texts"""
        # Convert all elements in texts to string and tokenize texts without transforming to ID
        tokens = [token for text in texts for token in self.token_pattern.findall(str(text).lower())]
        token_counts = Counter(tokens)
        self.token2id = {token: idx for idx, (token, _) in enumerate(token_counts.items())}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        self.save()

    def save(self):
        """Save token maps to disk"""
        with open(f"{self.path}/token2id.json", 'w') as f:
            json.dump(self.token2id, f)
        with open(f"{self.path}/id2token.json", 'w') as f:
            json.dump(self.id2token, f)

    def load(self) -> tuple[Any, Any]:
        """Load token maps from disk"""
        with open(f"{self.path}/token2id.json", 'r') as f:
            token2id = json.load(f)
        with open(f"{self.path}/id2token.json", 'r') as f:
            id2token = json.load(f)
        return token2id, id2token
