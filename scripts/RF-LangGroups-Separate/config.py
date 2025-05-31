
TRAINING_DATA_PATH = "../../data/train.csv"
TEST_DATA_PATH = "../../data/test.csv"
DATA_PATH = "../../data"

ONE_HOT_ENCODER_LANGUAGE = "../../resources"

    
NODE_FEATURES = [
    "eccentricity",
    "degree_cent",
    "harmoni_cent",
    "closeness_cent",
    "subgraph_cent",
    "current_flow_closeness",
    "katz_cent",
    "degree",
    "eigen_cent",
    "comm_betweenness",
    "second_order_cent",
    "avg_shortest_path_length",
    "vote_rank_score",
    "effective_size", ##
    "betweeness_cent",
    "largest_component_removed",
    "num_subtrees_removed",
    "subtree_size_variance",
    "participation_diameter",
    "radiality",
    "neighbor_degree_mean",
    "neighbor_degree_max",
    "neighbor_degree_min",
    "num_leaf_neighbors",
    "current_flow_betweenness",
    "load_centrality",
    "page_cent",
    "is_leaf"
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

TRAIN_DROP_COLS = ['sentence','is_root','node_number','language','language_group', 'num_subtrees_removed', 'number_of_nodes',
                   'graph_num_edges','neighbor_degree_mean', 'neighbor_degree_max', 'neighbor_degree_min','radiality',\
                   'graph_avg_path_length','graph_diameter','graph_density','graph_average_degree',\
                    'current_flow_closeness', 'comm_betweenness','current_flow_betweenness',\
                    'harmoni_cent', 'katz_cent','subgraph_cent','effective_size','second_order_cent',\
                     'num_leaf_neighbors', 'subtree_size_variance','avg_shortest_path_length','page_cent','load_centrality']
# avg_shortest_path_length highly correlated with eccentricty and radiality
# page_cent corr degree_cent
# load_centrality - betweeness_cent /comm_betweenness - betweeness_cent

TEST_DROP_COLS = ['sentence','node_number','language','language_group','id']




