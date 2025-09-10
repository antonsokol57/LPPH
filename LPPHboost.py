import numpy as np
import networkx as nx
import pandas as pd
import random
import matplotlib.pyplot as plt
from features import compute_edge_features_parallel, ListDigraph
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from typing import Dict, List, Tuple
from pathlib import Path


class GraphLinkPredictor:
    """A class for performing graph link prediction using XGBoost with persistent path homology features."""
    
    def __init__(self, random_seed: int = 0x57575757):
        self.random_seed = random_seed
        self.best_params = None
        np.random.seed(random_seed)
        random.seed(random_seed)
        
    def generate_negative_samples(self, graph: nx.DiGraph, num_negatives: int) -> List[Tuple[int, int]]:
        """Generate negative samples"""
        num_nodes = len(graph.nodes())
        existing_edges = set(graph.edges())
        neg_samples = []
        
        while len(neg_samples) < num_negatives:
            u, v = np.random.randint(0, num_nodes, size=2)
            if (u, v) not in existing_edges and u != v:
                neg_samples.append((u, v))
        return neg_samples
    
    def load_graph_data(self, dataset_name: str) -> nx.DiGraph:
        dataset_paths = {
            "bison": r"data_final\Bison\out.moreno_bison_bison",
            "highschool": r"data\moreno_highschool\out.moreno_highschool_highschool",
            "caenorhabditis": r"data_final\Caenorhabditis_elegans\out.dimacs10-celegansneural",
            "congress_vote": r"data_final\Congress_votes\out.convote",
            "florida_ecosystem_dry": r"data_final\Florida_ecosystem_dry\out.foodweb-baydry",
            "japanese_macaques": r"data_final\Japanese_macaques\out.moreno_mac_mac",
            "little_rock_lake": r"data_final\Little_Rock_Lake\out.maayan-foodweb",
            "physicians": r"data_final\Physicians\out.moreno_innovation_innovation",
            "figeys": r"data_final\Proteins-figeys\out.maayan-figeys",
            "stelzl": r"data_final\Proteins-Stelzl\out.maayan-Stelzl",
            "air_traffic_control": r"data_final\air_traffic_control\out.maayan-faa",
            "adolescent": r"data\moreno_health\out.moreno_health_health",
            "twitter": r"data\ego-twitter\out.ego-twitter",
            "google+": r"data\ego-gplus\out.ego-gplus",
            "dblp": r"data\dblp-cite\out.dblp-cite",
            "digg": r"data\munmun_digg_reply\out.munmun_digg_reply",
            "slashdot": r"data\slashdot-threads\out.slashdot-threads",
            "cora": r"data\subelj_cora\out.subelj_cora_cora",
            "PB": r"data_2\out.dimacs10-polblogs",
            "EMA": r"data_2\out.dnc-temporalGraph",
            "ATC": r"data_2\out.maayan-faa",
            "USA": r"data_2\out.opsahl-usairport",
            "CELE": r"data_2\out.dimacs10-celegans_metabolic"
        }
        
        if dataset_name not in dataset_paths:
            raise ValueError(f"Dataset {dataset_name} not found in available datasets")
            
        file_path = dataset_paths[dataset_name]
        
        # Define data format for each dataset
        data_formats = {
            "bison": [("weight", float)],
            "highschool": [("weight", float)],
            "caenorhabditis": [("weight", float)],
            "congress_vote": [("weight", float)],
            "florida_ecosystem_dry": [("weight", float)],
            "japanese_macaques": [("weight", float)],
            "little_rock_lake": [],
            "physicians": [],
            "figeys": [],
            "stelzl": [("weight", float)],
            "air_traffic_control": [],
            "adolescent": [("weight", float)],
            "twitter": [],
            "google+": [],
            "dblp": [("weight", float), ("time", float)],
            "digg": [("weight", float), ("time", float)],
            "slashdot": [("weight", float), ("time", float)],
            "cora": [],
            "PB": [],
            "EMA": [("weight", float), ("time", float)],
            "ATC": [],
            "USA": [("weight", float)],
            "CELE": []
        }
        
        return nx.read_edgelist(
            file_path,
            comments="%",
            data=data_formats[dataset_name],
            nodetype=int,
            create_using=nx.DiGraph,
        )
    
    def preprocess_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Preprocess graph by removing self-loops and relabeling nodes."""
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        return nx.relabel_nodes(graph, mapping)
    
    def compute_features(self, graph: ListDigraph, edges: List[Tuple[int, int]], 
                        max_depth: int, max_dim: int, max_workers: int = 12) -> np.ndarray:
        chunksize = len(edges) // 24 + 1
        features = compute_edge_features_parallel(
            graph, edges, max_depth=max_depth, resolution=10,
            max_dim=max_dim, max_workers=max_workers, chunksize=chunksize,
            total=len(edges)
        )
        return np.array(features)
    
    def train_and_evaluate(self, dataset_name: str, edge_depth: int = 5, dim: int = 3, 
                          n_trials: int = 30) -> List[float]:
        """Train and evaluate the model on the specified dataset."""
        auc_scores = []
        
        for trial in range(n_trials):
            current_seed = self.random_seed + trial
            np.random.seed(current_seed)
            random.seed(current_seed)
            
            print(f"Trial {trial+1}/{n_trials}: Computing {dataset_name} with edge depth {edge_depth}")

            nx_digraph = self.load_graph_data(dataset_name)
            nx_digraph = self.preprocess_graph(nx_digraph)

            pos_edges = list(nx_digraph.edges())
            neg_edges = self.generate_negative_samples(nx_digraph, len(pos_edges))

            all_edges = pos_edges + neg_edges
            all_labels = [1] * len(pos_edges) + [0] * len(neg_edges)

            train_edges, test_edges, train_labels, test_labels = train_test_split(
                all_edges, all_labels, test_size=0.1, random_state=current_seed, stratify=all_labels
            )
            
            # Create training graph
            train_graph = nx_digraph.copy()
            train_graph.remove_edges_from(test_edges)  

            edge_graph = ListDigraph(len(train_graph), len(train_graph.edges()))
            for u, v in train_graph.edges():
                edge_graph.add_edge(u, v)

            X_train = self.compute_features(edge_graph, train_edges, edge_depth, dim)
            X_test = self.compute_features(edge_graph, test_edges, edge_depth, dim)
            y_train, y_test = np.array(train_labels), np.array(test_labels)
            
            # Hyperparameter tuning (only on first trial)
            if trial == 0:
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [5, 7, 9],
                    "subsample": [0.9],
                }
                
                base_model = XGBClassifier(objective="binary:logistic", random_state=current_seed)
                grid_search = GridSearchCV(
                    estimator=base_model, param_grid=param_grid, 
                    scoring="roc_auc", cv=3, n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                self.best_params = grid_search.best_params_
                print(f"Best parameters found: {self.best_params}")
            
            # Train
            model = XGBClassifier(
                **self.best_params, objective="binary:logistic", random_state=current_seed
            )
            model.fit(X_train, y_train)
            
            # Evaluate 
            test_preds = model.predict_proba(X_test)[:, 1]
            test_preds_class = model.predict(X_test)
            
            auc = roc_auc_score(y_test, test_preds)
            auc_scores.append(auc)
            
            print(f"Trial {trial+1} Results:")
            print(f"  Test Accuracy: {accuracy_score(y_test, test_preds_class):.4f}")
            print(f"  Test Precision: {precision_score(y_test, test_preds_class):.4f}")
            print(f"  ROC-AUC Score: {auc:.4f}")
            print(f"  Mean AUC after {trial+1} trials: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.3f})")
            
            # Plot ROC curve for the first trial
            if trial == 0:
                fpr, tpr, _ = roc_curve(y_test, test_preds)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"XGBoost ROC Curve - {dataset_name}")
                plt.legend()
                plt.savefig(f"roc_curve_{dataset_name}.png")
                plt.close()
        
        return auc_scores


if __name__ == "__main__":
    predictor = GraphLinkPredictor()

    DATASETS = ["caenorhabditis"]
    EDGE_DEPTHS = [5]
    DIM = 3
    N_TRIALS = 30
    
    for dataset in DATASETS:
        for depth in EDGE_DEPTHS:
            print(f"\n=== Processing {dataset} with edge depth {depth} ===\n")
            auc_scores = predictor.train_and_evaluate(dataset, depth, DIM, N_TRIALS)
            
            print(f"\nFinal Results for {dataset} (depth {depth}):")
            print(f"Mean AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.3f})")
            print(f"Min AUC: {np.min(auc_scores):.4f}")
            print(f"Max AUC: {np.max(auc_scores):.4f}")