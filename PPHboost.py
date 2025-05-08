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


def generate_negative_samples(graph, num_negatives):
    num_nodes = len(graph.nodes())
    existing_edges = set(graph.edges())
    neg_samples = []

    while len(neg_samples) < num_negatives:
        u, v = np.random.randint(0, num_nodes, size=2)
        if (u, v) not in existing_edges and u != v:
            neg_samples.append((u, v))
    return neg_samples


if __name__ == "__main__":
    random_seed = 0x57575757
    np.random.seed(random_seed)
    random.seed(random_seed)
    for edge_depth in [5]:
        for name in [
            # "twitter"
            # "bison",
            # "highschool",
            "caenorhabditis",
            # "congress_vote",
            # "florida_ecosystem_dry",
            # "japanese_macaques",
            # "little_rock_lake",
            # "physicians",
            # "figeys",
            # "stelzl",
            # "air_traffic_control",
        ]:
            AUCS = []
            dim = 3
            best_params = None
            for i in range(30):

                random_seed += 1
                np.random.seed(random_seed)
                random.seed(random_seed)

                print(f"computing {name} with edge depth {edge_depth}")
                if name == "bison":
                    file_path = r"data_final\Bison\out.moreno_bison_bison"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "highschool":
                    file_path = (
                        r"data\moreno_highschool\out.moreno_highschool_highschool"
                    )
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "caenorhabditis":
                    file_path = (
                        r"data_final\Caenorhabditis_elegans\out.dimacs10-celegansneural"
                    )
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "congress_vote":
                    file_path = r"data_final\Congress_votes\out.convote"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "florida_ecosystem_dry":
                    file_path = r"data_final\Florida_ecosystem_dry\out.foodweb-baydry"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "japanese_macaques":
                    file_path = r"data_final\Japanese_macaques\out.moreno_mac_mac"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "little_rock_lake":
                    file_path = r"data_final\Little_Rock_Lake\out.maayan-foodweb"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "physicians":
                    file_path = (
                        r"data_final\Physicians\out.moreno_innovation_innovation"
                    )
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "figeys":
                    file_path = r"data_final\Proteins-figeys\out.maayan-figeys"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "stelzl":
                    file_path = r"data_final\Proteins-Stelzl\out.maayan-Stelzl"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "air_traffic_control":
                    file_path = r"data_final\air_traffic_control\out.maayan-faa"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "adolescent":
                    file_path = r"data\moreno_health\out.moreno_health_health"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "twitter":
                    file_path = r"data\ego-twitter\out.ego-twitter"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "google+":
                    file_path = r"data\ego-gplus\out.ego-gplus"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "dblp":
                    file_path = r"data\dblp-cite\out.dblp-cite"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float), ("time", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "digg":
                    file_path = r"data\munmun_digg_reply\out.munmun_digg_reply"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float), ("time", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "slashdot":
                    file_path = r"data\slashdot-threads\out.slashdot-threads"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float), ("time", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "cora":
                    file_path = r"data\subelj_cora\out.subelj_cora_cora"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "PB":
                    file_path = r"data_2\out.dimacs10-polblogs"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "EMA":
                    file_path = r"data_2\out.dnc-temporalGraph"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float), ("time", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "ATC":
                    file_path = r"data_2\out.maayan-faa"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "USA":
                    file_path = r"data_2\out.opsahl-usairport"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[("weight", float)],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )
                elif name == "CELE":
                    file_path = r"data_2\out.dimacs10-celegans_metabolic"
                    nx_digraph = nx.read_edgelist(
                        file_path,
                        comments="%",
                        data=[],
                        nodetype=int,
                        create_using=nx.DiGraph,
                    )

                mapping = {node: i for i, node in enumerate(nx_digraph.nodes())}
                nx_digraph.remove_edges_from(list(nx.selfloop_edges(nx_digraph)))
                nx_digraph = nx.relabel_nodes(nx_digraph, mapping)

                # Prepare training and testing data
                pos_edges = list(nx_digraph.edges())
                neg_edges = generate_negative_samples(nx_digraph, len(pos_edges))

                # Create features and labels
                edge_features = []
                labels = []

                # Split data before feature computation to avoid data leakage
                all_edges = pos_edges + neg_edges
                all_labels = [1] * len(pos_edges) + [0] * len(neg_edges)

                # Split data
                (train_edges, test_edges, train_labels, test_labels) = train_test_split(
                    all_edges,
                    all_labels,
                    test_size=0.1,
                    random_state=random_seed,
                    stratify=all_labels,
                )

                # Create training graph
                train_graph = nx_digraph.copy()
                train_graph.remove_edges_from(
                    test_edges
                )  # Remove only positive test edges

                # Compute features using training graph
                edge_graph = ListDigraph(len(train_graph), len(train_graph.edges()))
                for u, v in train_graph.edges():
                    edge_graph.add_edge(u, v)

                # Compute features
                train_features = compute_edge_features_parallel(
                    edge_graph,
                    train_edges,
                    max_depth=edge_depth,
                    resolution=10,
                    max_dim=dim,
                    max_workers=12,
                    chunksize=len(train_edges) // 24 + 1,
                    total=len(train_edges),
                )

                test_features = compute_edge_features_parallel(
                    edge_graph,
                    test_edges,
                    max_depth=edge_depth,
                    resolution=10,
                    max_dim=dim,
                    max_workers=12,
                    chunksize=len(test_edges) // 24 + 1,
                    total=len(test_edges),
                )

                # Convert to numpy arrays
                X_train = np.array(train_features)
                X_test = np.array(test_features)
                y_train = np.array(train_labels)
                y_test = np.array(test_labels)

                if i == 0:
                    # Hyperparameter tuning only on first iteration
                    param_grid = {
                        "n_estimators": [100, 200, 300],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [5, 7, 9],
                        "subsample": [0.9],
                    }

                    base_model = XGBClassifier(
                        objective="binary:logistic", random_state=random_seed
                    )

                    grid_search = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        scoring="roc_auc",
                        cv=3,
                        n_jobs=-1,
                    )
                    grid_search.fit(X_train, y_train)

                    best_params = grid_search.best_params_
                    print(f"Best parameters found: {best_params}")

                    model = XGBClassifier(
                        **best_params,
                        objective="binary:logistic",
                        random_state=random_seed,
                    )
                else:
                    # Reuse best parameters from first iteration
                    model = XGBClassifier(
                        **best_params,
                        objective="binary:logistic",
                        random_state=random_seed,
                    )

                model.fit(X_train, y_train)

                # Evaluate
                test_preds = model.predict_proba(X_test)[:, 1]  # Get probability scores
                test_preds_class = model.predict(X_test)

                print("\nEvaluation Results:")
                print(f"Test Accuracy: {accuracy_score(y_test, test_preds_class):.4f}")
                print(
                    f"Test Precision: {precision_score(y_test, test_preds_class):.4f}"
                )
                print(f"ROC-AUC Score: {roc_auc_score(y_test, test_preds):.4f}")
                AUCS.append(roc_auc_score(y_test, test_preds))
                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_test, test_preds)
                plt.figure(figsize=(8, 6))
                plt.plot(
                    fpr,
                    tpr,
                    label=f"ROC Curve (AUC = {roc_auc_score(y_test, test_preds):.4f})",
                )
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("XGBoost ROC Curve")
                plt.legend()
                print(
                    f"(For {name} with {edge_depth} in {i+1} trials: Mean AUC = {np.mean(AUCS):.4f} (±{np.std(AUCS):.3f})"
                )
    # plt.show()
