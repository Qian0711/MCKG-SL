# MCKG-SL: Multi-feature Cross-aggregation Synthetic Lethality Prediction Based on Knowledge Graph
In recent years, synthetic lethality (SL) has attracted much attention as a novel therapeutic concept for cancer. However, existing SL prediction algorithms have some limitations, such as high requirements for data quality and reliability, and incomplete understanding of complex biological system interaction networks. Recent studies have tried to improve the model by learning the feature representation of genes through knowledge graph, but they have ignored some information in the local association subgraph of genes, and have not learned enough information about the interactions between genes. 
To overcome these challenges, we propose a novel Knowledge Graph-based Synthetic Lethality model named MCKG-SL, which more comprehensively learns the interaction information between genes. First, we extract local association subgraph of gene pairs from the knowledge graph, to focus on the local association information around gene pairs. Then, we utilize Relational Graph Convolutional Network (RGCN) for global relational awareness and Graph Attention Network (GAT) for partial connection concern to learn the gene feature information in the subgraph. Subsequently, we design a multi-feature cross aggregation module to cross-fuse the relational features learned from the local association subgraph with biological features extracted from multi-omics data, enhancing the interactive learning of gene pair features. A large number of experimental results show that MCKG-SL method is superior to other advanced methods in SL prediction, and has strong generalization ability.


![model][model3.jpg]
