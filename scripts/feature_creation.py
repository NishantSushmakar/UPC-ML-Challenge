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
        
        return {v: (eccentricity[v],degree_cent[v], harmoni_cent[v], betweeness_cent[v], page_cent[v],eigen_cent[v],closeness_cent[v],\
                    katz_cent[v],information_cent[v],load_centrality[v],subgraph_cent[v],comm_betweenness[v],current_flow_closeness[v],\
                    current_flow_betweenness[v],second_order_cent[v],degree[v],\
                    avg_shortest_path_length[v],effective_size[v],vote_rank_score[v],is_leaf[v],\
                    largest_component_removed[v],num_subtrees_removed[v],subtree_size_variance[v],participation_diameter[v],\
                    radiality[v],neighbor_degree_mean[v],neighbor_degree_max[v],neighbor_degree_min[v],num_leaf_neighbors[v]) for v in T}
    

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

    

