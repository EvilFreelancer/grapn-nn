import requests
import json
import os

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": "token " + os.getenv('GITHUB_TOKEN'),
    "Accept": "application/vnd.github.v4+json, application/vnd.github.hawkgirl-preview+json"
}

base_query = """
{
  search(query: "type:user repos:>2 language:PHP", type: USER, first: 100, after: %s) {
    nodes {
      ... on User {
        login
      }
    }
    pageInfo {
      endCursor
      hasNextPage
    }
  }
}
"""

end_cursor = "null"
users = []

# Fetch users in batches of 100 until we reach 1000 or no more users
while len(users) < 1000:
    query = base_query % end_cursor
    response = requests.post(
        GITHUB_GRAPHQL_URL,
        headers=HEADERS,
        json={'query': query}
    )

    if response.status_code != 200:
        print("Error fetching users")
        break

    data = response.json()
    if "errors" in data:
        print(f"Error in GraphQL response: {data['errors']}")
        break

    batch_users = [node['login'] for node in data['data']['search']['nodes']]
    users.extend(batch_users)

    if not data['data']['search']['pageInfo']['hasNextPage']:
        break

    end_cursor = json.dumps(data['data']['search']['pageInfo']['endCursor'])

# Save to file
with open("gh_users.txt", "w") as file:
    for user in users:
        file.write(user + '\n')

print(f"Saved {len(users)} users to gh_users.txt")
