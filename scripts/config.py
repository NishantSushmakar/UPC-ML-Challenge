
TRAINING_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"
DATA_PATH = "../data"

ONE_HOT_ENCODER_LANGUAGE = "../resources"


NODE_FEATURES = [
    "eccentricity",
    "degree_cent",
    "harmoni_cent",
    "betweeness_cent",
    "page_cent",
    "eigen_cent",
    "closeness_cent",
    "katz_cent",
    "information_cent",
    "load_centrality",
    "subgraph_cent",
    "comm_betweenness",
    "current_flow_closeness",
    "current_flow_betweenness",
    "second_order_cent",
    "degree",
    "avg_shortest_path_length",
    "effective_size",
    "vote_rank_score",
    "is_leaf",
    "largest_component_removed",
    "num_subtrees_removed",
    "subtree_size_variance",
    "participation_diameter",
    "radiality",
    "neighbor_degree_mean",
    "neighbor_degree_max",
    "neighbor_degree_min",
    "num_leaf_neighbors"
]


LANGUAGE_TO_GROUP = {
    "English": "Germanic",
    "German": "Germanic",
    "Swedish": "Germanic",
    "Icelandic": "Germanic",
    "French": "Romance",
    "Spanish": "Romance",
    "Italian": "Romance",
    "Portuguese": "Romance",
    "Galician": "Romance",
    "Russian": "Slavic",
    "Polish": "Slavic",
    "Czech": "Slavic",
    "Finnish": "Uralic",
    "Turkish": "Turkic",
    "Hindi": "Indo-Aryan",
    "Japanese": "Japonic",
    "Korean": "Koreanic",
    "Chinese": "Sino-Tibetan",
    "Indonesian": "Austronesian",
    "Arabic": "Semitic",
    "Thai": "Kra-Dai"
}


TRAIN_DROP_COLS = ['sentence','is_root','node_number','language','language_group',
                   'graph_num_edges','neighbor_degree_mean','radiality',\
                   'graph_avg_path_length','graph_diameter','graph_density','graph_average_degree',\
                   'num_subtrees_removed','current_flow_betweenness','load_centrality','current_flow_closeness',\
                    'effective_size','num_subtrees_removed','comm_betweenness','page_cent','subgraph_cent',\
                    'avg_shortest_path_length','number_of_nodes','largest_component_removed','katz_cent',\
                    'closeness_cent','information_cent']
TEST_DROP_COLS = ['sentence','node_number','language','language_group','id']




