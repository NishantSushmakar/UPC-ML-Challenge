import pandas as pd
import networkx as nx
from collections import defaultdict

PATH = "../../data/"

# ====== Load Training Graphs ======
def load_train_data(train_path=f"{PATH}train.csv"):
    df_train = pd.read_csv(train_path)
    train_trees = defaultdict(list)
    for _, row in df_train.iterrows():
        G = nx.Graph(eval(row['edgelist']))
        train_trees[row['language']].append((G, row['root']))
    return train_trees

# ====== Node Similarity Heuristic ======
def find_most_similar_node(G_test, root_train, G_train):
    degree_train = G_train.degree(root_train)
    betweenness_train = nx.betweenness_centrality(G_train).get(root_train, 0)
    closeness_train = nx.closeness_centrality(G_train).get(root_train, 0)

    candidates = []
    for node in G_test.nodes():
        deg = G_test.degree(node)
        b = nx.betweenness_centrality(G_test).get(node, 0)
        c = nx.closeness_centrality(G_test).get(node, 0)

        dist = ((deg - degree_train) ** 2 +
                (b - betweenness_train) ** 2 +
                (c - closeness_train) ** 2)
        candidates.append((node, dist))

    return min(candidates, key=lambda x: x[1])[0]

# ====== Predict One Root ======
def predict_root(train_trees, test_edges, language):
    G_test = nx.Graph(test_edges)

    # Try edit distance matching
    best_score = float("inf")
    best_G_train, best_root = None, None
    for G_train, root in train_trees[language]:
        try:
            score = nx.graph_edit_distance(G_test, G_train)
            if score is not None and score < best_score:
                best_score = score
                best_G_train = G_train
                best_root = root
        except:
            continue

    if best_G_train is None:
        # fallback: use max degree if no match
        degrees = dict(G_test.degree())
        return max(degrees, key=degrees.get)

    return find_most_similar_node(G_test, best_root, best_G_train)

# ====== Predict for All Test Samples ======
def predict_test_set(train_trees, test_path=f"{PATH}test.csv", output_path=f"{PATH}predictions.csv"):
    df_test = pd.read_csv(test_path)
    predictions = []
    iso_match_count = 0

    for _, row in df_test.iterrows():
        lang = row['language']
        test_edges = eval(row['edgelist'])
        G_test = nx.Graph(test_edges)

        matched = False
        for G_train, root_train in train_trees[lang]:
            matcher = nx.algorithms.isomorphism.GraphMatcher(G_test, G_train)
            if matcher.is_isomorphic():
                phi = matcher.mapping
                inv_phi = {v: k for k, v in phi.items()}
                pred_root = inv_phi[root_train]
                matched = True
                iso_match_count += 1
                break

        if not matched:
            pred_root = predict_root(train_trees, test_edges, lang)

        predictions.append({'id': row['id'], 'root': pred_root})

    pd.DataFrame(predictions).to_csv(output_path, index=False)

    total = len(df_test)
    percent = iso_match_count / total * 100
    print(f"\nExact isomorphism matches: {iso_match_count}/{total} ({percent:.2f}%)")
    print(f"Predictions saved to: {output_path}")

# ====== Run Everything ======
if __name__ == "__main__":
    print("Loading training data...")
    train_trees = load_train_data()

    print("Predicting test roots...")
    predict_test_set(train_trees)
