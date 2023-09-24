import requests
import json
import os
from collections import defaultdict

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": "token " + os.getenv('GITHUB_TOKEN'),
    "Accept": "application/vnd.github.v4+json, application/vnd.github.hawkgirl-preview+json"
}


def fetch_repos_for_user(username, end_cursor=None):
    """Fetch repositories for a given user using GraphQL with pagination."""
    query = """
    {
        user(login: "%s") {
            repositories(first: 100, after: %s) {
                nodes {
                    name
                    languages(first: 50) {
                        edges {
                            size
                            node {
                                name
                            }
                        }
                        totalSize
                    }
                    repositoryTopics(first: 20) {
                        nodes {
                            topic {
                                name
                            }
                        }
                    }
                }
                pageInfo {
                    endCursor
                    hasNextPage
                }
            }
        }
    }
    """ % (username, json.dumps(end_cursor) if end_cursor else 'null')

    response = requests.post(
        GITHUB_GRAPHQL_URL,
        headers=HEADERS,
        json={'query': query}
    )

    if response.status_code != 200:
        print(f"Error fetching repos for {username} with end_cursor {end_cursor}")
        return []

    data = response.json()
    if "errors" in data:
        print(f"Error in GraphQL response: {data['errors']}")
        return []

    repos = data["data"]["user"]["repositories"]["nodes"]

    # If there's more data, fetch the next page
    if data["data"]["user"]["repositories"]["pageInfo"]["hasNextPage"]:
        end_cursor = data["data"]["user"]["repositories"]["pageInfo"]["endCursor"]
        return repos + fetch_repos_for_user(username, end_cursor)
    return repos


def build_graph_data_for_user(username, exclude_isolated=True):
    repos = fetch_repos_for_user(username)

    tech_weights = defaultdict(float)

    for repo in repos:

        # Programming languages and their weights
        total_lines = repo["languages"]["totalSize"]
        for language_data in repo["languages"]["edges"]:
            language_name = language_data["node"]["name"].lower()
            lines = language_data["size"]
            percentage = lines / total_lines
            tech_weights[language_name] += 5 + percentage  # correct for the total number of lines

        # Topics
        topics = [topic_data["topic"]["name"].lower() for topic_data in repo["repositoryTopics"]["nodes"]]
        for topic in topics:
            if topic in TECHNOLOGY_RELATIONS:
                tech_weights[topic] += 1

        # Dependencies
        # for manifest in repo["dependencyGraphManifests"]["nodes"]:
        #     for dependency in manifest["dependencies"]["nodes"]:
        #         package_name = dependency["packageName"].lower()
        #         if package_name in TECHNOLOGY_RELATIONS:
        #             tech_weights[package_name] += 0.5

    nodes = [{"id": tech} for tech in tech_weights.keys()]
    links = []
    for tech, related_techs in TECHNOLOGY_RELATIONS.items():
        for related_tech in related_techs:
            if tech in tech_weights and related_tech in tech_weights:
                links.append({
                    "source": tech,
                    "target": related_tech,
                    "weight": tech_weights[tech] + tech_weights[related_tech]
                })

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


def save_to_jsonl(data, filename="output_graphs.jsonl"):
    """Save data to a jsonl file."""
    with open(filename, 'a') as f:
        f.write(json.dumps(data))
        f.write('\n')


def main():
    with open('users.txt', 'r') as f:  # Список пользователей
        usernames = [line.strip() for line in f]

    for username in usernames:
        user_data = build_graph_data_for_user(username)
        save_to_jsonl(user_data)


if __name__ == "__main__":
    main()
