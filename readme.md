# Temporal Graph Neural Network

This project implements a temporal graph neural network (GNN) using the `TemporalGAT` architecture, which incorporates attention mechanisms for temporal and relational data. The pipeline supports advanced features like class balancing, graph augmentation, and multiple architectures.

---

## Features

### 1. Temporal Relations
- Handles temporal relations between nodes (past, present, future, distant past, and distant future).
- Uses `edge_type` to differentiate between temporal relationships.

### 2. Attention Mechanism
- Integrates Graph Attention Networks (GAT) to dynamically weight the importance of neighboring nodes.
- Supports multi-head attention for robust feature aggregation.

### 3. Regularization
- Dropout is applied between layers to prevent overfitting.
- L2 regularization (weight decay) is applied during optimization.

### 4. Early Stopping
- Monitors validation loss to stop training when overfitting occurs.

### 5. Graph Augmentation
- Adds noise to node features or edges for robustness.
- Optionally removes random edges (DropEdge).

### 6. Class Balancing
- Uses weighted cross-entropy loss to handle imbalanced datasets.

### 7. Multiple Architectures
- Implements a variety of models including:
  - Temporal GAT
  - Combined GCN+GAT
  - Combined GCN+RGCN
  - GAT with BatchNorm
  - GCN with Dropout

---

## Pipeline Overview

### **Data Preparation**
- Data is prepared in the form of a graph with:
  - `x`: Node features.
  - `y`: Node labels.
  - `edge_index`: Graph connectivity.
  - `edge_type`: Types of relationships (e.g., temporal or cross-modal).

### **Training and Evaluation**
The pipeline:
1. Computes class weights for balancing.
2. Trains models with:
   - Adam optimizer.
   - Weighted cross-entropy loss.
   - Dropout and weight decay.
3. Implements early stopping to prevent overfitting.
4. Evaluates models using accuracy and F1-score.

---

## Key Functions

### `TemporalGAT`
The main model leveraging temporal and relational attention.


### `train_and_evaluate`
Trains and evaluates the model, supporting models with `edge_type`.

---

## Usage

1. Prepare your data in the expected format (`train_data`, `val_data`, `test_data`).
2. Initialize the model and pipeline.
3. Train, evaluate, and benchmark.

---

## Improvements to Consider

1. **Graph Augmentation**:
   - Add noise to node features or randomly drop edges.

2. **Advanced Architectures**:
   - Implement GATv2 or GraphSAGE for improved performance.

3. **Hyperparameter Tuning**:
   - Adjust learning rates, dropout probabilities, and hidden dimensions.

4. **Cross-Validation**:
   - Evaluate the model more robustly with k-fold cross-validation.

---
