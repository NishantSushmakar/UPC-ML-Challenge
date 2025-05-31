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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class LanguageFeature(BaseEstimator,TransformerMixin):

    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    
    def fit_transform(self,df,y=None):
        print("Langauge Feature Started")
        df['family_group'] = df['language'].apply(lambda x:config.LANGUAGE_TO_GROUP[x])
        print("Langauge Feature Ended")
        return df
    
    def transform(self,df):
        print("Langauge Feature Started")
        df['family_group'] = df['language'].apply(lambda x:config.LANGUAGE_TO_GROUP[x])
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

    def normalize_sentence_features(self, features_dict):
        """
        Normalize all centrality features at the sentence level using MinMax scaling
        """
        # Convert features dict to numpy array
        nodes = list(features_dict.keys())
        features = np.array([list(features_dict[node]) for node in nodes])
        
        # Initialize scaler
        scaler = MinMaxScaler()
        
        # Normalize all features (except non-centrality features at the end)
        # We'll normalize the first 15 features which are centrality measures
        num_centrality_features = len(config.NODE_FEATURES)
        features[:, :num_centrality_features] = scaler.fit_transform(features[:, :num_centrality_features])
        
        # Convert back to dictionary format
        normalized_features = {}
        for i, node in enumerate(nodes):
            normalized_features[node] = tuple(features[i])
        
        return normalized_features

    
    def create_node_features(self,edgelist):

        T = nx.from_edgelist(edgelist)

        eccentricity = nx.eccentricity(T)

        degree_cent = nx.degree_centrality(T)
        harmoni_cent = nx.harmonic_centrality(T)
        betweeness_cent = nx.betweenness_centrality(G=T,normalized=True)
        page_cent = nx.pagerank(T)
        eigen_cent = nx.eigenvector_centrality(T, max_iter=10000)
        closeness_cent = nx.closeness_centrality(T)
        katz_cent = nx.katz_centrality(T,max_iter=10000)
        # information_cent = nx.information_centrality(T)
        load_centrality = nx.load_centrality(G=T,normalized=True)
        subgraph_cent = nx.subgraph_centrality(T)
        comm_betweenness = nx.communicability_betweenness_centrality(T)
        current_flow_closeness = nx.current_flow_closeness_centrality(T)
        current_flow_betweenness = nx.current_flow_betweenness_centrality(T,normalized=True)
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
        
    
        features_dict = {v: (eccentricity[v],degree_cent[v], harmoni_cent[v], betweeness_cent[v], page_cent[v],eigen_cent[v],closeness_cent[v],\
                    katz_cent[v],load_centrality[v],subgraph_cent[v],comm_betweenness[v],current_flow_closeness[v],\
                    current_flow_betweenness[v],second_order_cent[v],degree[v],\
                    avg_shortest_path_length[v],effective_size[v],vote_rank_score[v],is_leaf[v],\
                    largest_component_removed[v],num_subtrees_removed[v],subtree_size_variance[v],participation_diameter[v],\
                    radiality[v],neighbor_degree_mean[v],neighbor_degree_max[v],neighbor_degree_min[v],num_leaf_neighbors[v]) for v in T}

        # Normalize the features at sentence level
        normalized_features = self.normalize_sentence_features(features_dict)
        
        return normalized_features

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
        sample_df['structural_group'] = lst['structural_group']  

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
        df = df.reset_index(drop=True)
        df = self.graph_features_extract(df)
        X = df.drop(columns=['edgelist','root'])
        y = df['root'].values
        print("XXXXXXXXX")
        print(X)

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


class StructuralLanguageGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None  # Will be initialized in fit()
        self.language_to_group = {}
        
    def extract_language_features(self, df):
        """Group features by language and compute averages"""
        lang_features = defaultdict(list)
        
        for _, row in df.iterrows():
            features = self._extract_sentence_features(row['edgelist'])
            lang_features[row['language']].append(features)
        
        X, langs = [], []
        for lang, all_features in lang_features.items():
            X.append(np.mean(all_features, axis=0))
            langs.append(lang)
            
        return np.array(X), langs

    def _extract_sentence_features(self, edgelist):
        """Helper: Extract features for a single sentence"""
        G = nx.from_edgelist(ast.literal_eval(edgelist))
        return [
            nx.density(G),
            len(G.edges()) / len(G.nodes()),
            nx.average_shortest_path_length(G) if nx.is_connected(G) else 0,
            len(list(nx.connected_components(G)))
        ]

    def fit(self, df, y=None):
        # 1. Get language-level averaged features
        X, languages = self.extract_language_features(df)
        
        # 2. Determine actual number of clusters needed
        n_unique_languages = len(languages)
        actual_clusters = min(n_unique_languages, self.n_clusters)
        
        if n_unique_languages < self.n_clusters:
            print(f"Warning: Only {n_unique_languages} languages available. "
                  f"Reducing clusters from {self.n_clusters} to {actual_clusters}")
        
        # 3. Initialize and fit KMeans with appropriate cluster count
        self.kmeans = KMeans(n_clusters=actual_clusters, random_state=self.random_state)
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        
        # 4. Create language to group mapping
        self.language_to_group = {
            lang: f"structural_group_{label}" 
            for lang, label in zip(languages, self.kmeans.labels_)
        }
        
        # 5. Save mapping
        self.save_mapping()
        
        # 6. Print group assignments
        self.print_group_assignments()
        
        return self
    
    def print_group_assignments(self):
        """Print which languages belong to which groups"""
        group_to_langs = defaultdict(list)
        for lang, group in self.language_to_group.items():
            group_to_langs[group].append(lang)
        
        print("\nStructural Group Assignments:")
        for group, langs in sorted(group_to_langs.items()):
            print(f"{group}: {', '.join(sorted(langs))}")
    
    def save_mapping(self):
        """Save the language to group mapping to a file"""
        mapping_path = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, "language_to_structural_group.pkl")
        joblib.dump(self.language_to_group, mapping_path)
        
    def load_mapping(self):
        """Load the language to group mapping from a file"""
        mapping_path = os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, "language_to_structural_group.pkl")
        if os.path.exists(mapping_path):
            self.language_to_group = joblib.load(mapping_path)
            return True
        return False
    
    def transform(self, X):
        X = X.copy()
        # If we haven't fit (like in test time), try to load the mapping
        if not self.language_to_group and not self.load_mapping():
            raise ValueError("No language to group mapping found. Fit the transformer first.")
            
        X['structural_group'] = X['language'].map(self.language_to_group)
        return X
    
    def get_group_mapping(self):
        """Return dictionary of {group_name: [list_of_languages]}"""
        group_to_langs = defaultdict(list)
        for lang, group in self.language_to_group.items():
            group_to_langs[group].append(lang)
        return dict(group_to_langs)

class LanguageOHE(BaseEstimator,TransformerMixin):
    def __init__(self, enc_lan, enc_lan_family, enc_structural_group):
        self.enc_lan = enc_lan
        self.enc_lan_family = enc_lan_family
        self.enc_structural_group = enc_structural_group
        
    def fit(self, X, y=None):
        return self

    def _get_clean_structural_names(self, encoder):
        """Generate clean column names like structural_group_0, structural_group_1"""
        return [f"structural_group_{i}" for i in range(len(encoder.categories_[0]))]

    def fit_transform(self, df, y=None):
        print("One Hot Encoding Started")
        # Language encoder
        encoder = OneHotEncoder()
        encoded_array = encoder.fit_transform(df[['language']])
        encoded_df = pd.DataFrame(encoded_array.toarray(), 
                               columns=encoder.get_feature_names_out(['language']))

        # Family encoder
        family_encoder = OneHotEncoder()
        encoded_array_family = family_encoder.fit_transform(df[['language_group']])
        encoded_family_df = pd.DataFrame(encoded_array_family.toarray(),
                                      columns=family_encoder.get_feature_names_out(['language_group']))

        # Structural group encoder with clean names
        structural_encoder = OneHotEncoder()
        encoded_array_structural = structural_encoder.fit_transform(df[['structural_group']])
        
        # Use clean column names
        structural_columns = self._get_clean_structural_names(structural_encoder)
        encoded_structural_df = pd.DataFrame(encoded_array_structural.toarray(),
                                          columns=structural_columns)

        final_df = pd.concat([df, encoded_df, encoded_family_df, encoded_structural_df], axis=1)
        
        # Save encoders
        joblib.dump(encoder, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, self.enc_lan))
        joblib.dump(family_encoder, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, self.enc_lan_family))
        joblib.dump(structural_encoder, os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, self.enc_structural_group))

        print("One Hot Encoding created and Saved")
        return final_df

    def transform(self, df):
        print("One Hot Encoding Started")
        # Load encoders
        encoder = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, self.enc_lan))
        family_encoder = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, self.enc_lan_family))
        structural_encoder = joblib.load(os.path.join(config.ONE_HOT_ENCODER_LANGUAGE, self.enc_structural_group))

        # Transform with consistent column names
        encoded_array = encoder.transform(df[['language']])
        encoded_df = pd.DataFrame(encoded_array.toarray(), 
                               columns=encoder.get_feature_names_out(['language']))
        
        encoded_array_family = family_encoder.transform(df[['language_group']])
        encoded_family_df = pd.DataFrame(encoded_array_family.toarray(),
                                      columns=family_encoder.get_feature_names_out(['language_group']))

        encoded_array_structural = structural_encoder.transform(df[['structural_group']])
        structural_columns = self._get_clean_structural_names(structural_encoder)
        encoded_structural_df = pd.DataFrame(encoded_array_structural.toarray(),
                                          columns=structural_columns)

        final_df = pd.concat([df, encoded_df, encoded_family_df, encoded_structural_df], axis=1)
       
        print("One Hot Encoding created and Saved")
        return final_df
    
class LanguageOHEOG(BaseEstimator,TransformerMixin):

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


