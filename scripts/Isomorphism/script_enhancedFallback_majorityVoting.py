import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from collections import defaultdict, Counter
import numpy as np

PATH = "../../data/"

# ====== Step 1: Load and Preprocess Training Data ======
def load_train_data(train_path=f"{PATH}train.csv"):
    df_train = pd.read_csv(train_path)
    train_trees = defaultdict(list)
    for _, row in df_train.iterrows():
        G = nx.Graph(eval(row['edgelist']))
        root = row['root']
        train_trees[row['language']].append((G, root))
    return train_trees

# ====== Step 2: Enhanced Prediction with Majority Voting and Fallback ======
def enhanced_fallback(G):
    """Combines multiple centrality measures for robust fallback prediction"""
    try:
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        degree = dict(G.degree())
        
        # Attempt eigenvector centrality (might fail for disconnected graphs)
        try:
            eigen = nx.eigenvector_centrality(G, max_iter=1000)
            max_eigen = max(eigen.values())
            eigen = {k: v/max_eigen for k, v in eigen.items()}  # Normalize
        except:
            eigen = {n: 0 for n in G.nodes}
        
        # Combine scores with weights
        combined_scores = {
            n: (0.40 * betweenness[n] +
                0.20 * closeness[n] +
                0.20 * degree[n] +
                0.20 * eigen[n])
            for n in G.nodes
        }
        
        return max(combined_scores, key=combined_scores.get)
    
    except Exception as e:
        # Fallback to simple degree centrality if something fails
        degrees = dict(G.degree())
        return max(degrees, key=degrees.get)

def predict_root(train_trees, test_edges, language, iso_counter):
    G_test = nx.Graph(test_edges)
    candidate_roots = []
    
    # Check same-language matches
    for G_train, root in train_trees.get(language, []):
        if nx.is_isomorphic(G_test, G_train):
            matcher = nx.algorithms.isomorphism.GraphMatcher(G_test, G_train)
            if matcher.is_isomorphic():
                phi = matcher.mapping
                inv_phi = {v: k for k, v in phi.items()}
                candidate_roots.append(inv_phi[root])
    
    # Majority voting for isomorphic matches
    if candidate_roots:
        iso_counter[0] += 1
        most_common = Counter(candidate_roots).most_common(1)
        return most_common[0][0], True
    
    # Enhanced fallback prediction
    fallback_root = enhanced_fallback(G_test)
    return fallback_root, False

# ====== Step 3: Modified Prediction Pipeline ======
def predict_test_set(train_trees, test_path=f"{PATH}test.csv", output_path=f"{PATH}predictions.csv"):
    df_test = pd.read_csv(test_path)
    predictions = []
    iso_match_count = [0]

    for _, row in df_test.iterrows():
        lang = row['language']
        test_edges = eval(row['edgelist'])
        root_pred, used_iso = predict_root(train_trees, test_edges, lang, iso_match_count)
        predictions.append({
            'id': row['id'],
            'root_pred': root_pred,
            'used_iso': used_iso
        })

    df_preds = pd.DataFrame(predictions)
    df_preds.to_csv(output_path, index=False)
    total = len(df_test)
    print(f"Predictions saved to {output_path}")
    print(f"Isomorphism matches: {iso_match_count[0]}/{total} ({iso_match_count[0]/total:.2%})")
    return df_preds

# ====== Execution ======
if __name__ == "__main__":
    print("Loading training data...")
    train_trees = load_train_data()
    print("Predicting test set...")
    predictions = predict_test_set(train_trees)
