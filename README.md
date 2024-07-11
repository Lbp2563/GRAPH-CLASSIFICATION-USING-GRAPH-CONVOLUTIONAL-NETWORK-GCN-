# Graph Classification using Graph Convolutional Network (GCN)

## Introduction
This project implements a Graph Convolutional Network (GCN) using PyTorch Geometric for graph classification. The model is trained on the MUTAG dataset, which consists of chemical compounds labeled according to their mutagenicity.

## Dataset
The MUTAG dataset is a collection of nitroaromatic compounds aimed at predicting their mutagenicity on *Salmonella typhimurium*. The dataset includes 188 samples of chemical compounds with 7 discrete node labels. The dataset files include:

1. **DS_A.txt**: Sparse (block diagonal) adjacency matrix for all graphs.
2. **DS_graph_indicator.txt**: Column vector of graph identifiers for all nodes of all graphs.
3. **DS_graph_labels.txt**: Class labels for all graphs in the dataset.
4. **DS_node_labels.txt**: Column vector of node labels.

Optional files, if available:
- **DS_edge_labels.txt**: Labels for the edges.
- **DS_edge_attributes.txt**: Attributes for the edges.
- **DS_node_attributes.txt**: Matrix of node attributes.
- **DS_graph_attributes.txt**: Regression values for all graphs in the dataset.

### Node Labels
- 0: Carbon (C)
- 1: Nitrogen (N)
- 2: Oxygen (O)
- 3: Fluorine (F)
- 4: Iodine (I)
- 5: Chlorine (Cl)
- 6: Bromine (Br)

### Edge Labels
- 0: Aromatic
- 1: Single
- 2: Double
- 3: Triple

## Model Architecture
The Graph Convolutional Network (GCN) is designed with the following components:
1. Input: Node features, edge index, and batch information.
2. Layers:
   - Three Graph Convolutional layers with ReLU activation function.
   - Global Mean Pooling layer.
   - Dropout layer (p=0.5).
   - Linear classifier.

### Algorithm
The GCN algorithm operates on graph-structured data:
1. Input node features, edge indices, and batch information.
2. Perform graph convolution using multiple GCN layers.
3. Aggregate node-level features into a fixed-size representation for each graph using global pooling.
4. Apply a linear classifier to predict the output class probabilities.

## Training
- **Optimizer**: Adam optimizer with learning rate 0.01.
- **Loss Function**: Cross-Entropy Loss.
- **Procedure**:
  - Iterate over the training dataset in batches.
  - Perform forward pass.
  - Compute loss.
  - Backpropagate gradients.
  - Update parameters.

## Testing
- Evaluate the model on both training and test datasets.
- Compute accuracy as the ratio of correct predictions to the total number of samples.

## Code Structure
1. **Data Loading**:
   - Load MUTAG dataset using PyTorch Geometric.
   - Split the dataset into training and test sets.
2. **Model Definition**:
   - Define the GCN model architecture using PyTorch Geometric.
3. **Training and Evaluation**:
   - Train the model using the training dataset.
   - Evaluate the model on both training and test datasets.
   - Display training and test accuracies for each epoch.

## Results
- **Training Accuracy**: 0.7933
- **Test Accuracy**: 0.7632

## References
Debnath, A. K., Lopez de Compadre, R. L., Debnath, G., Shusterman, A. J., & Hansch, C. (1991). Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. Correlation with molecular orbital energies and hydrophobicity. *Journal of Medicinal Chemistry*, 34(2), 786-797.

## Dataset Link
[MUTAG Dataset](https://paperswithcode.com/dataset/mutag)
