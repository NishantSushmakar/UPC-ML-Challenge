import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import config
import networkx as nx
import numpy as np
import os
import ast
from sklearn.preprocessing import OneHotEncoder
import joblib
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class LanguageFeature(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    
    def fit_transform(self,df,y=None):
        print("Langauge Feature Started")
        df['family_group'] = df['language'].apply(lambda x:config.LANGUAGE_TO_GROUP[x])
        df['sov_order'] = df['language'].apply(lambda x: 1 if x in config.DOMINANT_ORDER_GROUPED['SOV'] else 0)
        df['svo_order'] = df['language'].apply(lambda x: 1 if x in config.DOMINANT_ORDER_GROUPED['SVO'] else 0)
        df['vso_order'] = df['language'].apply(lambda x: 1 if x in config.DOMINANT_ORDER_GROUPED['VSO'] else 0)
        print("Langauge Feature Ended")
        return df
    
    def transform(self,df):
        print("Langauge Feature Started")
        df['family_group'] = df['language'].apply(lambda x:config.LANGUAGE_TO_GROUP[x])
        df['sov_order'] = df['language'].apply(lambda x: 1 if x in config.DOMINANT_ORDER_GROUPED['SOV'] else 0)
        df['svo_order'] = df['language'].apply(lambda x: 1 if x in config.DOMINANT_ORDER_GROUPED['SVO'] else 0)
        df['vso_order'] = df['language'].apply(lambda x: 1 if x in config.DOMINANT_ORDER_GROUPED['VSO'] else 0)
        print("Langauge Feature Ended")
        return df


class GraphFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def create_graph_features(self,edgelist):
        
        T = nx.from_edgelist(edgelist)
    
        avg_path_length = nx.average_shortest_path_length(T)
        diameter = nx.diameter(T)
        num_edges = T.number_of_edges()
        density = nx.density(T)

        n = T.number_of_nodes()
        average_degree = (2 * num_edges) / n


        return {
                "avg_path_length": avg_path_length,
                "diameter": diameter,
                "num_edges": num_edges,
                "density": density,
                "average_degree": average_degree
                }
    
    def fit(self,X,y=None):
        return self
    
    def fit_transform(self,df,y=None):
        df['edgelist'] = df['edgelist'].apply(ast.literal_eval)
        print("Graph Features Creation Started")
        df['graph_features'] = df['edgelist'].apply(lambda x:self.create_graph_features(x))
        print("Graph Feature Creation Ended")
        return df
    
    def transform(self,df):
        df['edgelist'] = df['edgelist'].apply(ast.literal_eval)
        print("Graph Features Creation Started")
        df['graph_features'] = df['edgelist'].apply(lambda x:self.create_graph_features(x))
        print("Graph Feature Creation Ended")
        return df
    
class NodeFeatures(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass

    def leaf_removal_steps(self,G):
        G = G.copy()  
        removal_step = {}
        step = 0

        while G.nodes:
            
            leaves = [node for node in G.nodes if G.degree(node) == 1]
            
            if not leaves:
                # If no leaves but graph still has nodes, it's likely a cycle or fully connected component.
                # Remove all remaining nodes in next step.
                leaves = list(G.nodes)
            
            # Record the current step for all leaves
            for leaf in leaves:
                removal_step[leaf] = step

            # Remove all leaf nodes
            G.remove_nodes_from(leaves)
            step += 1

        return removal_step
    
    def max_height_from_each_node(self,G):
        heights = {}
        for node in G.nodes:
            # Use BFS to compute shortest paths from node to all others
            lengths = nx.single_source_shortest_path_length(G, node)
            heights[node] = max(lengths.values())
        return heights
    

    def compute_tree_features(self,G):
    
        nodes = list(G.nodes())
        features = {}
        
        for node in nodes:
            # Calculate distances from this node to all other nodes
            distances = nx.single_source_shortest_path_length(G, node)
            
            # Find leaf nodes (nodes with degree 1, except the potential root if it has degree 1)
            leaf_nodes = [n for n in G.nodes() if G.degree[n] == 1 and n != node]
            
            # If the node itself is a leaf, we still want to compute its features
            if not leaf_nodes and G.degree[node] == 1:
                # In this case, the only leaf node is the node itself, but we exclude it
                # so we need to find the second most distant node
                distances_without_self = {k: v for k, v in distances.items() if k != node}
                max_distance = max(distances_without_self.values()) if distances_without_self else 0
                leaf_distances = [0]  # Only for computation purposes
            else:
                # Distance to leaf nodes
                leaf_distances = [distances[leaf] for leaf in leaf_nodes]
                max_distance = max(distances.values()) if distances else 0
            
            # Calculate subtree features
            subtree_features = self.calculate_subtree_features(G, node)
            
            # Combine all features
            features[node] = {
                # Distance-based features
                'avg_distance_to_all_nodes': np.mean(list(distances.values())),
                'max_distance_to_any_node': max_distance,
                'avg_distance_to_leaf_nodes': np.mean(leaf_distances) if leaf_distances else 0,
                'std_distance_to_leaf_nodes': np.std(leaf_distances) if len(leaf_distances) > 1 else 0,
                'distance_to_farthest_leaf': max(leaf_distances) if leaf_distances else 0,
                
                # Add subtree features
                **subtree_features
            }
        
        return features

    def calculate_subtree_features(self, G, root):
        """Calculate subtree features for a given root node in graph G."""
        # Create directed BFS tree rooted at 'root'
        T = nx.bfs_tree(G, root)
        
        # Function to compute subtree sizes via DFS
        def get_subtree_sizes(T, root_node):
            sizes = {}
            def dfs(node):
                size = 1  # Include the node itself
                for child in T.successors(node):
                    size += dfs(child)
                sizes[node] = size
                return size
            dfs(root_node)
            return sizes
        
        # Get subtree sizes for all nodes
        subtree_sizes = get_subtree_sizes(T, root)
        children = list(T.successors(root))  # Immediate children of the root
        
        # Edge case: no children (root is a leaf)
        if not children:
            return {
                'max_subtree_size': 0,
                'min_subtree_size': 0,
                'subtree_size_balance': 0,
                'num_subtrees': 0,
                'subtree_size_std': 0,
                'max_subtree_depth': 0,
                'min_subtree_depth': 0,
                'subtree_depth_balance': 0,
                'subtree_depth_std': 0
            }
        
        # Compute depths of all nodes relative to the original root
        depths = nx.shortest_path_length(T, root)
        
        # Calculate max depth for each child's subtree (relative to the child)
        subtree_depths = []
        for child in children:
            # Get all nodes in the subtree rooted at 'child'
            subtree_nodes = nx.descendants(T, child)
            subtree_nodes.add(child)  # Include the child itself
            
            if not subtree_nodes:
                # Subtree has only the child (depth = 0)
                max_depth = 0
            else:
                # Compute max depth within this subtree (relative to child)
                max_depth = max(depths[n] - depths[child] for n in subtree_nodes)
            
            subtree_depths.append(max_depth)
        
        # Subtree sizes of immediate children
        children_subtree_sizes = [subtree_sizes[child] for child in children]
        
        return {
            # Subtree size features
            'max_subtree_size': max(children_subtree_sizes),
            'min_subtree_size': min(children_subtree_sizes),
            'subtree_size_balance': max(children_subtree_sizes) - min(children_subtree_sizes),
            'num_subtrees': len(children),
            'subtree_size_std': np.std(children_subtree_sizes) if len(children_subtree_sizes) > 1 else 0,
            
            # Subtree depth features
            'max_subtree_depth': max(subtree_depths) if subtree_depths else 0,
            'min_subtree_depth': min(subtree_depths) if subtree_depths else 0,
            'subtree_depth_balance': max(subtree_depths) - min(subtree_depths) if subtree_depths else 0,
            'subtree_depth_std': np.std(subtree_depths) if len(subtree_depths) > 1 else 0
        }
        
    def find_centroids(self,G):
        T = G.copy()
        leaves = [node for node in T.nodes if T.degree(node) == 1]
        while len(T.nodes) > 2:
            T.remove_nodes_from(leaves)
            leaves = [node for node in T.nodes if T.degree(node) == 1]
        return list(T.nodes)
    

    def calculate_centrality_scores(self,G, positions=None):
        """
        Calculate spatial centrality scores (D, C, and D') for each node in a graph.
        
        Parameters:
        - edgelist: List of tuples representing edges (source, target)
        - positions: Dictionary mapping nodes to their positions in linear arrangement
                    If None, positions will be assigned based on node order in the graph
        
        Returns:
        - Dictionary containing D, C, and D' scores for each node
        """
        
        # If positions are not provided, assign sequential positions
        if positions is None:
            nodes = sorted(G.nodes())
            positions = {node: i for i, node in enumerate(nodes)}
        
        n = len(positions)  # Number of nodes
        
        # Calculate scores for each node
        scores = {}
        for node in G.nodes():
            # Get node's neighbors
            neighbors = list(G.neighbors(node))
            
            # Calculate D(v) - sum of distances to neighbors
            D_v = sum(abs(positions[neighbor] - positions[node]) for neighbor in neighbors)
            
            # Calculate C(v) - coverage
            # Include the node itself with its neighbors
            extended_positions = [positions[node]] + [positions[neighbor] for neighbor in neighbors]
            C_v = max(extended_positions) - min(extended_positions)
            
            # Calculate D'(v) - corrected spatial centrality
            D_prime_v = (C_v / (n - 1)) * D_v if n > 1 else 0
            
            # Store scores
            scores[node] = {
                "D": D_v,
                "C": C_v,
                "D_prime": D_prime_v
            }
        
        return scores

    def create_node_features(self,edgelist):

        T = nx.from_edgelist(edgelist)

        eccentricity = nx.eccentricity(T)

        degree_cent = nx.degree_centrality(T)
        harmoni_cent = nx.harmonic_centrality(T)
        betweeness_cent = nx.betweenness_centrality(T)
        page_cent = nx.pagerank(T)
        eigen_cent = nx.eigenvector_centrality(T, max_iter=10000)
        closeness_cent = nx.closeness_centrality(T)
        katz_cent = nx.katz_centrality(T,max_iter=10000)
        information_cent = nx.information_centrality(T)
        load_centrality = nx.load_centrality(T)
        subgraph_cent = nx.subgraph_centrality(T)
        comm_betweenness = nx.communicability_betweenness_centrality(T)
        current_flow_closeness = nx.current_flow_closeness_centrality(T)
        current_flow_betweenness = nx.current_flow_betweenness_centrality(T)
        second_order_cent = nx.second_order_centrality(T)

        degree = T.degree()
        
        avg_shortest_path_length = {}
        for node in T.nodes():
            avg_path_length = sum(nx.shortest_path_length(T, source=node).values()) / (len(T) - 1)
            avg_shortest_path_length[node] = avg_path_length


        effective_size = nx.effective_size(T)

        vote_rank = nx.voterank(T)
        vote_rank_score ={}
        for i, node in enumerate(vote_rank):
            vote_rank_score[node] = len(T) - i
        for node in T.nodes():
            if node not in vote_rank_score:
                    vote_rank_score[node] = 0



        is_leaf = {node: (T.degree[node] == 1) for node in T.nodes} 


        largest_component_removed = {}
        for node in T.nodes:
            H = T.copy()
            H.remove_node(node)
            largest_cc = max(nx.connected_components(H), key=len, default=set())
            largest_component_removed[node] = len(largest_cc)  


        num_subtrees_removed = dict(T.degree())

        subtree_size_variance = {}
        for node in T.nodes:
            H = T.copy()
            H.remove_node(node)
            components = list(nx.connected_components(H))
            sizes = [len(c) for c in components]
            subtree_size_variance[node] = np.var(sizes) if sizes else 0.0 

        
        d = nx.diameter(T)
        participating_nodes = set()
        # Find all nodes in any diameter path
        for u in T.nodes:
            for v in T.nodes:
                if u < v and nx.shortest_path_length(T, u, v) == d:
                    path = nx.shortest_path(T, u, v)
                    participating_nodes.update(path)

        participation_diameter = {node: (node in participating_nodes) for node in T.nodes}

        radiality = {}
        for node in T.nodes:
            path_lengths = nx.single_source_shortest_path_length(T, node)
            sum_d = sum(path_lengths.values())
            avg_d = sum_d / (len(path_lengths) - 1) if len(path_lengths) > 1 else 0.0
            radiality[node] = eccentricity[node] - avg_d

        neighbor_degree_mean, neighbor_degree_max, neighbor_degree_min = {}, {}, {}
        for node in T.nodes:
            degrees = [T.degree[neigh] for neigh in T.neighbors(node)]
            if not degrees:
                neighbor_degree_mean[node] = neighbor_degree_max[node] = neighbor_degree_min[node] = 0
            else:
                neighbor_degree_mean[node] = np.mean(degrees)
                neighbor_degree_max[node] = np.max(degrees)
                neighbor_degree_min[node] = np.min(degrees)


        num_leaf_neighbors = {}
        for node in T.nodes:
            num_leaf_neighbors[node] = sum(1 for neigh in T.neighbors(node) if T.degree[neigh] == 1)

        steps = self.leaf_removal_steps(T)
        max_heights = self.max_height_from_each_node(T)

        tree_based_features = self.compute_tree_features(T)

        centroids = self.find_centroids(T)
        centroid_dict = {}
        for node in T.nodes():
            centroid_dict[node] = int(node in centroids)

        new_cent_scores = self.calculate_centrality_scores(T)

        
        return {v: (eccentricity[v],degree_cent[v], harmoni_cent[v], betweeness_cent[v], page_cent[v],eigen_cent[v],closeness_cent[v],\
                    katz_cent[v],information_cent[v],load_centrality[v],subgraph_cent[v],comm_betweenness[v],current_flow_closeness[v],\
                    current_flow_betweenness[v],second_order_cent[v],degree[v],\
                    avg_shortest_path_length[v],effective_size[v],vote_rank_score[v],is_leaf[v],\
                    largest_component_removed[v],num_subtrees_removed[v],subtree_size_variance[v],participation_diameter[v],\
                    radiality[v],neighbor_degree_mean[v],neighbor_degree_max[v],neighbor_degree_min[v],num_leaf_neighbors[v],steps[v],max_heights[v],\
                    tree_based_features[v]['avg_distance_to_all_nodes'],tree_based_features[v]['max_distance_to_any_node'],tree_based_features[v]['avg_distance_to_leaf_nodes'],\
                    tree_based_features[v]['std_distance_to_leaf_nodes'],tree_based_features[v]['distance_to_farthest_leaf'],  tree_based_features[v]['max_subtree_size'],\
                    tree_based_features[v]['min_subtree_size'], tree_based_features[v]['subtree_size_balance'],   tree_based_features[v]['num_subtrees'],\
                    tree_based_features[v]['subtree_size_std'],tree_based_features[v]['max_subtree_depth'],tree_based_features[v]['min_subtree_depth'],\
                    tree_based_features[v]['subtree_depth_balance'],tree_based_features[v]['subtree_depth_std'],centroid_dict[v],new_cent_scores[v]["D"],\
                    new_cent_scores[v]['C'],new_cent_scores[v]['D_prime']) for v in T}
    

    def fit(self,X,y=None):
        return self
    
    def fit_transform(self,df,y=None):
        print("Node Features Creation Started")
        df['node_features'] = df['edgelist'].apply(lambda x:self.create_node_features(x))
        print("Node Features Creation Ended")
        return df

    def transform(self,df):
        print("Node Features Creation Started")
        df['node_features'] = df['edgelist'].apply(lambda x:self.create_node_features(x))
        print("Node Features Creation Ended")
        return df
    

class FormatDataFrame(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass

    def graph_features_extract(self,df):
        graph_df = pd.json_normalize(df['graph_features']).add_prefix('graph_')
        df = pd.concat([df.drop('graph_features', axis=1), graph_df], axis=1)

        return df

    def extract_features(self,lst,y=None,check=True):

        sample_df =  pd.DataFrame.from_dict(lst['node_features'],orient='index',columns=config.NODE_FEATURES)
        sample_df['node_number'] = sample_df.index
        sample_df['sentence'] = lst['sentence']    
        sample_df['language'] = lst['language']
        sample_df['language_group'] = lst['family_group']
        sample_df['number_of_nodes'] = lst['n']
        sample_df['graph_avg_path_length'] = lst['graph_avg_path_length']
        sample_df['graph_diameter'] = lst['graph_diameter']
        sample_df['graph_num_edges'] = lst['graph_num_edges']
        sample_df['graph_density'] = lst['graph_density']
        sample_df['graph_average_degree'] = lst['graph_average_degree']

        if check and ('id' in lst.keys()) :
            sample_df['id'] = lst['id']
        
        if not check or ('id' not in lst.keys()):
            sample_df['is_root'] = 0
            sample_df.loc[sample_df['node_number'] == y, 'is_root'] = 1

    
        return sample_df
    
    def create_data(self,X,y=None,check=True):

        final_df = []


        if check and (y is not None):
            for i in range(X.shape[0]):
                final_df.append(self.extract_features(lst=X.loc[i],y=y[i],check=check))
        elif check:
            for i in range(X.shape[0]):
                final_df.append(self.extract_features(lst=X.loc[i],check=check))
        else:
            for i in range(X.shape[0]):
                final_df.append(self.extract_features(lst=X.loc[i],y=y[i],check=check))

        return pd.concat(final_df).reset_index(drop=True)
    

    
    def fit(self,X,y=None):
        return self


    def fit_transform(self,df,y=None):
        print("DataFrame Creation Started")
        df = self.graph_features_extract(df)
        X = df.drop(columns=['edgelist','root'])
        y = df['root'].values
        main_data = self.create_data(X,y,False)
        print("DataFrame Creation Ended!!")
        return main_data
    
    def transform(self,df):
        print("DataFrame Creation Started")
        df = self.graph_features_extract(df)
        if 'root' in df.columns:
            
            X = df.drop(columns=['edgelist','root']) 
            y = df['root'].values
            main_data = self.create_data(X,y=y,check=True)
        else:
            X = df.drop(columns=['edgelist']) 
            main_data = self.create_data(X,y=None,check=True)
        print("DataFrame Creation Ended!!")

        return main_data

class LanguageOHE(BaseEstimator,TransformerMixin):

    def __init__(self,enc_lan,enc_lan_family):
        self.enc_lan = enc_lan
        self.enc_lan_family = enc_lan_family
        

    def fit(self,X,y=None):
        return self

    def fit_transform(self, df,y=None):
        
        print("One Hot Encoding Started")
        encoder = OneHotEncoder()
        encoded_array = encoder.fit_transform(df[['language']])
        encoded_df = pd.DataFrame(encoded_array.toarray(), columns=encoder.get_feature_names_out())

        family_encoder = OneHotEncoder()
        encoded_array_familiy = family_encoder.fit_transform(df[['language_group']])
        encoded_family_df = pd.DataFrame(encoded_array_familiy.toarray(),columns=family_encoder.get_feature_names_out())

        final_df = pd.concat([df,encoded_df,encoded_family_df],axis=1)
        

        joblib.dump(encoder, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,self.enc_lan))
        joblib.dump(family_encoder, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,self.enc_lan_family))

        print("One Hot Encoding created and Saved")

        return final_df


    def transform(self,df):


        encoder = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,self.enc_lan))
        family_encoder = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,self.enc_lan_family))

        print("One Hot Encoding Started")
        encoded_array = encoder.transform(df[['language']])
        encoded_df = pd.DataFrame(encoded_array.toarray(), columns=encoder.get_feature_names_out())
        
        encoded_array_family = family_encoder.transform(df[['language_group']])
        encoded_family_df = pd.DataFrame(encoded_array_family.toarray(),columns=family_encoder.get_feature_names_out())

        final_df = pd.concat([df,encoded_df,encoded_family_df],axis=1)
       
        print("One Hot Encoding created and Saved")


        return final_df


class UnsupervisedModel(BaseEstimator,TransformerMixin):

    def __init__(self,model_path,scaler_path):
        self.model_path = model_path
        self.scaler_path = scaler_path


    def fit(self,x,y):
        return self
    
    def fit_transform(self,df,y=None):
        scaler = MinMaxScaler() 
        x_scaled = scaler.fit_transform(df.drop(columns=['language','language_group','is_root','sentence']))
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = model.fit_predict(x_scaled)
        df['cluster'] = labels

        joblib.dump(model,os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,self.model_path))
        joblib.dump(scaler,os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,self.scaler_path))
        return df


    def transform(self,df): 


        model = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,self.model_path))
        scaler = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE,self.scaler_path))

        x_scaled = scaler.transform(df.drop(columns=['language','language_group','is_root','sentence']))
        labels = model.fit_predict(x_scaled)

        df['cluster'] = labels

        return df





# def main():

#     df = pd.read_csv(config.TEST_DATA_PATH)

#     feature_pipeline = Pipeline(steps=[
#                 ("Language Features",LanguageFeature()),
#                 ("Graph Features",GraphFeatures()),
#                 ("Node Features",NodeFeatures()),
#                 ("Dataset Creation",FormatDataFrame()),
#                 ("Language One Hot Encoding",LanguageOHE(enc_lan="lan_encoder.pkl",enc_lan_family="lan_family_encoder.pkl"))  
#             ])
    
#     main_df = feature_pipeline.transform(df)
#     main_df.to_csv(os.path.join(config.DATA_PATH,"test_sample_df.csv"),index=False)

# if __name__ == "__main__":

#     main()

    

