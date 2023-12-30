from mapping.replacements import *


def exclude_isolated_nodes(nodes, links):
    connected_techs = set()
    for link in links:
        connected_techs.add(link["source"])
        connected_techs.add(link["target"])
    return [node for node in nodes if node["id"] in connected_techs]


def build_from_list(tech_list, relations, exclude_isolated=False):
    for to_remove in REMOVE:
        if to_remove in tech_list:
            tech_list.remove(to_remove)
    for replace_to, replacements in REPLACEMENTS.items():
        for replacement in replacements:
            if replacement in tech_list:
                tech_list.remove(replacement)
                tech_list.append(replace_to)
    nodes = [{"id": tech} for tech in tech_list]
    links = []
    added_pairs = set()
    for tech, related_techs in relations.items():
        if tech in tech_list and tech not in REMOVE:
            for related_tech in related_techs:
                if related_tech in tech_list:
                    # Formate pairs in a specific order for convenience
                    pair = tuple(sorted([tech, related_tech]))
                    # Check if the pair is already added
                    if pair not in added_pairs:
                        links.append({"source": tech,"target": related_tech,"weight": 0.01})
                        added_pairs.add(pair)
    # Exclude nodes without links if the option is enabled
    if exclude_isolated:
        nodes = exclude_isolated_nodes(nodes, links)
    return {"nodes": nodes, "links": links}
