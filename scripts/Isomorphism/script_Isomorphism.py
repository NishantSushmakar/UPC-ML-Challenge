import pandas as pd
import networkx as nx
from collections import defaultdict

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

# ====== Step 2: Predict Root for a Test Tree, return also if isomorphism used ======
def predict_root(train_trees, test_edges, language, iso_counter, fallback=True):
    G_test = nx.Graph(test_edges)
    
    # Try matching in the same language first
    for (G_train, root) in train_trees.get(language, []):
        if nx.is_isomorphic(G_test, G_train):
            matcher = nx.algorithms.isomorphism.GraphMatcher(G_test, G_train)
            if matcher.is_isomorphic():
                iso_counter[0] += 1
                phi = matcher.mapping
                inv_phi = {v: k for k, v in phi.items()}
                return inv_phi[root], True  # Isomorphism used
    
    # Fallback prediction: highest degree node
    if fallback:
        degrees = dict(G_test.degree())
        return max(degrees.keys(), key=lambda x: degrees[x]), False  # Fallback used
    
    return None, False

# ====== Step 3: Predict on Test Set and Save Predictions ======
def predict_test_set(train_trees, test_path=f"{PATH}test.csv", output_path=f"{PATH}predictions.csv"):
    df_test = pd.read_csv(test_path)
    predictions = []
    iso_match_count = [0]  # Mutable counter for isomorphism matches

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
    print(f"Exact isomorphism matches: {iso_match_count[0]}/{total} ({(iso_match_count[0]/total)*100:.2f}%)")


# ====== Run the pipeline ======
if __name__ == "__main__":
    print("Loading training data...")
    train_trees = load_train_data(f"{PATH}train.csv")
    print("Predicting test data...")
    predict_test_set(train_trees, f"{PATH}test.csv", f"{PATH}predictions.csv")
