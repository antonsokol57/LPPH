# Local Persistent Path Homology (LPPH) for Directed Link Prediction

This repository contains the official Python implementation of **Local Persistent Path Homology (LPPH)**, a novel topological feature vector designed for directed link prediction in graphs. The method is introduced in the thesis: 

> **Enhancing Machine Learning Algorithms on Directed Graphs Using Persistent Path Homology**  
> Sokolov Anton  
> Supervisor: Cand. Sc. Viktor A. Zamaraev

LPPH leverages **Persistent Path Homology (PPH)** to capture multi-scale directional topological features around graph edges. These features are transformed into fixed-dimensional vectors using **persistence images**, making them suitable for machine learning models.

---

## 🧠 What is LPPH?

LPPH is a topological feature extraction method that:

- Constructs an **edge-centric filtration** around a target directed edge.
- Computes **persistent path homology** to track the evolution of directed paths and cycles.
- Converts persistence diagrams into **persistence images**—stable, vectorized representations.
- Can be used **standalone** with classifiers like XGBoost or **integrated with GNNs** to enhance link prediction.

---

## ✨ Key Features

- 🧩 **Direction-Aware**: Captures asymmetric connectivity patterns in directed graphs.
- 📐 **Multi-Scale**: Extracts features at different filtration depths (e.g., 5, 9, 13, 17).
- 🔬 **Topologically Rich**: Encodes cycles, paths, and connectivity structures up to a chosen homology dimension.
- ⚡ **Efficient**: Parallelized computation using PHAT for boundary matrix reduction.
- 🤖 **Model-Agnostic**: Works with XGBoost, GNNs, or other classifiers.

---

## 📦 Installation

### Dependencies

- Python 3.8+
- `numpy`
- `networkx`
- `scikit-learn`
- `xgboost`
- `torch` (for GNN models)
- `torch_geometric` (for GNN layers)
- `phat` (for persistent homology computation)
- `persim` (for persistence images)
- `tqdm`

### Install via pip

```bash
pip install numpy networkx scikit-learn xgboost torch torch_geometric phat-persistence persim tqdm
```

---

## 🚀 Usage

### 1. Standalone XGBoost with LPPH Features

```python
from LPPHboost import GraphLinkPredictor

predictor = GraphLinkPredictor(random_seed=0x57575757)
auc_scores = predictor.train_and_evaluate(
    dataset_name="caenorhabditis",
    edge_depth=5,
    dim=3,
    n_trials=30
)
```

### 2. Integrating LPPH with Directed GNNs

See `modelsPPH.ipynb` for full examples using:

- `DirGCNConv`
- `DirSageConv`
- `DirGATConv`

Example snippet:

```python
from features import compute_edge_features_parallel, ListDigraph
from models import LinkPredictionModel

# Compute LPPH features for edges
edge_features = compute_edge_features_parallel(
    graph, edges, max_depth=5, resolution=10, max_dim=3
)

# Use in a GNN model
model = LinkPredictionModel(
    type='dirGAT',
    in_feats=in_dim,
    gnn_hidden_size=hidden_dim,
    edge_feature_use=True,
    edge_emb_size=edge_feat_dim,
    fc_hidden_size=128,
    num_hidden_layers=2,
    gnn_layers=2
)
```

---

## 📊 Datasets

We evaluate on 11 real-world directed graphs:

| Dataset                | Nodes | Edges | Domain         |
|------------------------|-------|-------|----------------|
| Bison                  | 26    | 314   | Social         |
| Highschool             | 70    | 366   | Social         |
| Caenorhabditis         | 297   | 4296  | Biological     |
| Congress Vote          | 219   | 764   | Organizational |
| Florida Ecosystem Dry  | 128   | 2137  | Ecological     |
| Japanese Macaques      | 62    | 1187  | Social         |
| Little Rock Lake       | 183   | 2494  | Ecological     |
| Physicians             | 241   | 1098  | Organizational |
| Figeys                 | 2239  | 6452  | Biological     |
| Stelzl                 | 1706  | 6207  | Biological     |
| Air Traffic Control    | 1226  | 2615  | Transportation |

---

## 📈 Results

LPPH improves link prediction performance:

- ✅ Boosts dirGNNs in **10/11** datasets
- ✅ Outperforms hybrid models in **8/11** datasets as a standalone XGBoost classifier
- ✅ Competitive with state-of-the-art methods like SRTGCN and IRW/DRW

Example AUC improvements:

| Dataset           | dirGAT (no LPPH) | dirGAT + LPPH | LPPH + XGBoost |
|-------------------|------------------|---------------|----------------|
| Highschool        | 0.7224           | 0.8252        | 0.8689         |
| Air Traffic Control | 0.7014         | 0.7430        | 0.8508         |
| Stelzl            | 0.7565           | 0.8878        | 0.9652         |

---

## 🧪 How to Reproduce

1. Clone the repo:
   ```bash
   git clone https://github.com/antonsoko157/LPPH.git
   cd LPPH
   ```

2. Run the Jupyter notebook for GNN experiments:
   ```bash
   jupyter notebook modelsPPH.ipynb
   ```

3. Use `LPPHboost.py` for XGBoost-based experiments.

4. Datasets are included in `data/` and `data_final/` directories.

---

## 🛠️ Code Structure

- `LPPHboost.py`: XGBoost classifier using LPPH features.
- `modelsPPH.ipynb`: Jupyter notebook with GNN models and experiments.
- `features.py`: Core functions for computing LPPH features.
- `data/`, `data_final/`: Dataset directories.

---

## 📮 Contact

For questions or collaborations, feel free to reach out:

- **Author**: Sokolov Anton  
- **Email**: [Your Email]  
- **GitHub**: [antonsoko157](https://github.com/antonsoko157)

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

**LPPH**: Bridging topology and machine learning for better directed graph analysis.

📜 License
This project is licensed under the MIT License. See LICENSE for details.

LPPH: Bridging topology and machine learning for better directed graph analysis.
