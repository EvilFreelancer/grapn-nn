from collections import defaultdict


def build_from_list(tech_list, relations, exclude_isolated=False):
    tech_weights = defaultdict(float)
    nodes = [{"id": tech, "weight": tech_weights[tech]} for tech in tech_list]
    links = []
    added_pairs = set()

    for tech, related_techs in relations.items():
        if tech in tech_list:
            for related_tech in related_techs:
                if related_tech in tech_list:
                    # Formate pairs in a specific order for convenience
                    pair = tuple(sorted([tech, related_tech]))
                    # Check if the pair is already added
                    if pair not in added_pairs:
                        links.append({
                            "source": tech,
                            "target": related_tech,
                            "weight": 1
                        })
                        added_pairs.add(pair)

    # Exclude nodes without links if the option is enabled
    if exclude_isolated:
        connected_techs = set()
        for link in links:
            connected_techs.add(link["source"])
            connected_techs.add(link["target"])
        nodes = [node for node in nodes if node["id"] in connected_techs]

    return {
        "nodes": nodes,
        "links": links
    }
