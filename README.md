## Local Persistent Path Homology (PPH) for Link Prediction

## Overview
This repository implements Local Persistent Path Homology (PPH) methods for link prediction in directed graphs. The project contains two main components:
1. **PPHboost.py** - XGBoost classifier using PPH features
2. **modelsPPH.ipynb** - GNN-based models (DirGCN, DirSage, DirGAT) with PPH feature integration

## Key Features
- Computes persistent homology features for edges in directed graphs
- Supports multiple GNN architectures with directional awareness
- Compares performance with and without PPH features
- Includes statistical significance testing
- Works with various real-world graph datasets

## Installation
```bash
git clone [your-repo-url]
cd [your-repo-name]
pip install -r requirements.txt
Requirements
Python 3.7+

PyTorch

NetworkX

XGBoost

scikit-learn

node2vec

Other standard scientific computing libraries

Usage
PPHboost.py
bash
python PPHboost.py
modelsPPH.ipynb
Open the Jupyter notebook and run cells sequentially:

bash
jupyter notebook modelsPPH.ipynb
Code Structure
text
├── PPHboost.py              # XGBoost implementation with PPH features
├── modelsPPH.ipynb          # GNN models with PPH integration
├── features.py              # Feature computation utilities (imported)
├── data/                    # Graph datasets
└── README.md
Key Improvements Made
Modularization: Separated graph loading into reusable functions

Configuration: Added central configuration management

Error Handling: Improved edge cases and weight validation

Documentation: Added comprehensive docstrings and comments

Reproducibility: Enhanced seed management across experiments

Results
The implementation shows comparative performance between:

Traditional GNN approaches

GNN + PPH feature augmentation

XGBoost with PPH features

Statistical tests (paired t-tests) determine significance of performance differences.

Contributing
Fork the repository

Create a feature branch

Make your changes

Add tests if applicable

Submit a pull request

Citation
If you use this code in your research, please cite:
[Your publication details here]

License
[Your chosen license]

Contact
[Your email/contact information]
