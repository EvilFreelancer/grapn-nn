import json
from crawler.sources.github import build_graph_data_for_user


def main():
    with open('gh_users.txt', 'r') as f:
        usernames = [line.strip() for line in f]

    with open("gh_output_graphs.jsonl", 'w') as f:
        for username in usernames:
            user_data = build_graph_data_for_user(username)
            f.write(json.dumps(user_data))
            f.write('\n')


if __name__ == "__main__":
    main()
