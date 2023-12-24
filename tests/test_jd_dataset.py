import unittest
from simple_gnn.dataset.jd_dataset import *


class TestFunctions(unittest.TestCase):

    def test_load_json(self):
        self.assertEqual(load_json("./tests/sample.json"), {"key": "value"})
        with self.assertRaises(FileNotFoundError):
            load_json("notexist.json")
        url = "https://raw.githubusercontent.com/ZhongTr0n/JD_Analysis/main/jd_data2.json"
        response = urlopen(url).read()
        self.assertEqual(load_json(url), json.loads(response))

    def test_load_jda(self):
        self.assertIsNotNone(load_jda("jd_data2.json"))
        with self.assertRaises(Exception):
            load_jda("invalid_url")

    def test_get_nodes(self):
        graph_data = {
            'nodes': [{'id': 'node1'}, {'id': 'node2'}, {'id': 'node3'}],
            'links': [{'source': 'node1', 'target': 'node2', 'value': 1.1},
                      {'source': 'node2', 'target': 'node3', 'value': 2.2}]
        }
        expected_output = (
            ['node1', 'node2', 'node3'],
            {0: 'node1', 1: 'node2', 2: 'node3'},
            {'node1': 0, 'node2': 1, 'node3': 2},
            [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]]
        )
        self.assertEqual(get_nodes(graph_data), expected_output)

    def test_get_edges(self):
        node_mapping = {'node1': 0, 'node2': 1, 'node3': 2}
        graph_data = {
            'nodes': [{'id': 'node1'}, {'id': 'node2'}, {'id': 'node3'}],
            'links': [{'source': 'node1', 'target': 'node2', 'value': 1.1},
                      {'source': 'node2', 'target': 'node3', 'value': 2.2}]
        }
        expected_output = [[0, 1], [1, 2]], [1.1, 2.2]
        self.assertEqual(get_edges(node_mapping, graph_data), expected_output)
