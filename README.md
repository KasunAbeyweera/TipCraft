# Tip Recommendation System

## 1. Introduction
This GitHub repository contains the implementation of a Tip Recommendation System, which employs two deep learning architectures: Autoencoder with K-means clustering and the Deep Embedding Clustering (DEC) algorithm. The primary objective is to create personalized tips for users by leveraging clustering techniques and incorporating vector storage of user cluster representations. The project introduces the retrieval-augmented generation (RAG) technique, enhancing the tip generation process.

## 2. High-Level Architecture
![High-Level Architecture](https://github.com/KasunAbeyweera/TipCraft/raw/main/notebooks/High-Level%20Architecture.png)
The system's architecture involves clustering users based on their characteristics and storing their vector representations in a database. The retrieval-augmented generation (RAG) technique is then used to aggregate reviews and tips from similar users within a cluster, providing personalized tips to users.

## 3. Data Selection
The Yelp user dataset was chosen for model training. To optimize computational efficiency, we selected limited features from the dataset. Both review and tip datasets were incorporated for tip recommendations, ensuring exposure to a broad spectrum of user behaviors and general learning experiences.

## 4. Model Selection
Two deep-learning clustering algorithms were chosen: Autoencoder with K-means clustering and Deep Embedding Clustering (DEC).

## 5. Methodology
### Data Collection
Data was extracted from the Yelp Academic Dataset User JSON file and converted into a structured Pandas DataFrame. Preprocessing steps, including feature transformation and engineering, were applied to prepare the data for model training. Duplicate rows and unnecessary columns were removed during data cleaning.

### Exploratory Data Analysis (EDA)
Descriptive statistics, histograms, and correlation matrices were utilized for exploratory data analysis, providing insights into feature distributions and relationships.

### Model 1: Autoencoder with K-Means
- The autoencoder model was implemented with K-means clustering.
- StandardScaler was used for data scaling, and the optimal number of clusters was determined using the elbow method.
- K-means clustering algorithm was applied, and cluster centers were visualized.
- Principal Component Analysis (PCA) was used for dimensionality reduction, visualizing clusters in 2D and 3D.

### Model 2: Deep Embedded Clustering (DEC)
- StandardScaler was applied to scale the dataset, and the optimal number of clusters was determined.
- A DEC model was implemented, including an encoder, decoder, and clustering layer.
- Autoencoder training and clustering layer addition were performed for unsupervised clustering.
- PCA was applied to visualize clusters in reduced dimensions.

### Vector Store and Retrieval-Augmented Generation (RAG)
Pinecone, a vector database service, was chosen to index user cluster representations obtained from deep learning models. Encoded vectors reflecting user characteristics were stored on Pinecone, serving as a foundation for implementing the retrieval-augmented generation (RAG) technique. OpenAI's gpt-4-1106-preview was used for the large language model.

## 6. Model Evaluation and Comparison
### Autoencoder Model
- Architecture:
  - Input Layer: 9 neurons
  - Encoding Layers: 500, 500, 2000 neurons with ReLU activation
  - Encoded Layer: 10 neurons with ReLU activation
  - Decoding Layers: 2000, 500 neurons with ReLU activation
  - Output Layer: 9 neurons
- Training:
  - Optimizer: Adam
  - Loss Function: Mean Squared Error (MSE)
  - Epochs: 5
  - Loss: 0.02
- Clustering (K-Means):
  - Used encoded features for clustering.
  - Explored different cluster numbers (1 to 8) and selected the optimal number based on the elbow plot.

### DEC Model
- Encoder Architecture:
  - Input Layer: 9 neurons
  - Hidden Layers: 128, 64 neurons with ReLU activation
  - Clustering Layer: 4 neurons
- Training:
  - Used Mean Squared Error (MSE) and Kullback-Leibler Divergence (KLD) as loss functions.
  - Trained the model for 20 epochs.
  - Loss: 0.1217
