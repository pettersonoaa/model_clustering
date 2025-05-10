import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import colormaps
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AffinityPropagation, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA 

# --- Configuration ---
OUTPUT_FILES = "output_files" # Directory to save output files (ensure it exists)
if not os.path.exists(OUTPUT_FILES):
    os.makedirs(OUTPUT_FILES)
    print(f"Created directory: {OUTPUT_FILES}")

CLUSTER_COLORS = {
    0: 'blue',
    1: 'orange',
    2: 'green',
    3: 'red',
    4: 'brown',
    5: 'pink',
    6: 'gray',
    7: 'olive',
    8: 'cyan',
    9: 'purple',
    10: 'magenta',
}


def fit_model(X, n_clusters: int, n_init=3, sample_size=None, verbose=0):

    if verbose > 0:
        print(f"\n--- Fitting model with {n_clusters} clusters ---")

    sample_data = X.copy()
    if sample_size is not None and len(X) > sample_size:
        if verbose > 0:
            print(f"Original dataset size: {X.shape[0]}")
            print(f"Sampling {sample_size} points from {X.shape[0]}...")
        sample_indices = np.random.choice(sample_data.shape[0], sample_size, replace=False)
        sample_data = sample_data[sample_indices, :]
    
    scaler = StandardScaler()
    preprocessor = Pipeline([('scaler', scaler)])
    sample_processed = preprocessor.fit_transform(sample_data)

    if verbose > 0:
        print(f"Sample size after processing: {sample_processed.shape[0]}")
        print(f"Sample shape: {sample_processed.shape}")
        print(f"Fitting model...")
    start_time = time.time()
    model = KMeans(
        n_clusters=n_clusters, 
        n_init=n_init, 
        random_state=42
    )
    labels = model.fit_predict(sample_processed)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose > -1:
        print(f"Model fitted. Time taken: {elapsed_time:.2f} seconds")
    if verbose > 0:
        print(f"Labels shape: {labels.shape}")
        print(f"Labels unique values: {np.unique(labels)}")
        print(f"Labels counts: {np.bincount(labels)}")
    
    return {
        "model": model,
        "labels": labels,
        "data": sample_processed,
    }

def find_best_k_elbow(model_name, X, max_k=10, sample_size=None, show=False):
    """
    Plot the Elbow Method and Silhouette Score in a single subplot.

    Parameters:
    - X (ndarray): The dataset (scaled or unscaled).
    - max_k (int): The maximum number of clusters to evaluate.
    """
    print(f"\n--- Visualizing Elbow and Silhouette ---")

    scaler = StandardScaler()
    preprocessor = Pipeline([('scaler', scaler)])
    X_processed = preprocessor.fit_transform(X)

    
    inertias = []
    silhouette_scores = []
    for k in range(2, max_k + 1):
        print(f"Scoring {k-1}/{max_k-1} models...")
        start_time = time.time()
        results = fit_model(X=X, n_clusters=k, sample_size=sample_size,verbose=0)
        inertias.append(results['model'].inertia_)
        silhouette_scores.append(silhouette_score(results['data'], results['labels']))
        elapsed_time = time.time() - start_time
        print(f"Scoring complete. Time taken: {elapsed_time:.2f} seconds")

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Elbow Method Plot
    axes[0].plot(range(2, max_k + 1), inertias, marker='o', color='blue')
    axes[0].set_title("Elbow Method for Optimal K")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia (Sum of Squared Distances)")
    axes[0].grid(True)

    # Silhouette Score Plot
    axes[1].plot(range(2, max_k + 1), silhouette_scores, marker='o', color='green')
    axes[1].set_title("Silhouette Score for Optimal K")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_file_name = f"{OUTPUT_FILES}/elbow_method_{model_name}.png"
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file_name}")

    if show:
        plt.show()

def visualize_clustering_results(
        model_name: str,  
        data_for_viz, 
        labels_for_viz,
        show=False
    ):
    """
    Visualize clustering results for a given model using PCA.

    Parameters:
    - model_name (str): The name of the model to visualize.
    - results (dict): The dictionary containing clustering results.
    - models (dict): The dictionary of clustering models.
    - X_processed_metrics_sample (ndarray): The metric sample used for visualization.
    """
    print(f"\n--- Visualizing results for {model_name} ---")

    pca = PCA(n_components=2, random_state=42)
    data_2d = pca.fit_transform(data_for_viz)

    plt.figure(figsize=(10, 8))

    # Use CLUSTER_COLORS to assign colors to clusters
    unique_labels_viz = np.unique(labels_for_viz)
    for label in unique_labels_viz:
        cluster_color = CLUSTER_COLORS.get(label, 'gray')  # Default to 'gray' if label not in CLUSTER_COLORS
        plt.scatter(data_2d[labels_for_viz == label, 0],
                    data_2d[labels_for_viz == label, 1],
                    s=30,
                    color=cluster_color,
                    label=f'Cluster {label}',
                    alpha=0.7)

    plt.title(f'Clustering Visualization for {model_name} (PCA-reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    if len(unique_labels_viz) < 15:  # Avoid overly crowded legend
        plt.legend()
    plt.grid(True)

    # Save the figure
    output_file_name = f"{OUTPUT_FILES}/clustering_visualization_{model_name}.png"
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file_name}")

    if show:
        plt.show()




# Assuming `X_processed` is the scaled dataset and `labels` are the cluster labels
def analyze_cluster_features(
        model_name: str, 
        X, 
        labels, 
        feature_names=None, 
        title="Cluster Feature Heatmap",
        show=False
    ):
    """
    Analyze which features explain each cluster by calculating feature means per cluster.
    Plot a heatmap to visualize feature means for each cluster.

    Parameters:
    - X (ndarray): The dataset (scaled or unscaled).
    - labels (ndarray): Cluster labels for each data point.
    - feature_names (list): Optional list of feature names.
    - title (str): Title of the heatmap.

    Returns:
    - cluster_feature_means (DataFrame): Mean values of features for each cluster.
    """
    print(f"\n--- Visualizing clusters by features ---")

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(X, columns=feature_names if feature_names else [f"Feature_{i}" for i in range(X.shape[1])])
    df['Cluster'] = labels

    # Calculate mean values for each feature per cluster
    cluster_feature_means = df.groupby('Cluster').mean()

    # print("\nFeature means per cluster:")
    # print(cluster_feature_means)

    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_feature_means, annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Clusters")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the figure
    output_file_name = f"{OUTPUT_FILES}/feature_heatmap_{model_name}.png"
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file_name}")

    if show:
        plt.show()

    return cluster_feature_means

def clustering_pipeline(
        X, 
        model_name="KMeans", 
        n_clusters=4, 
        sample_size=None, 
        max_k=10, 
        elbow_analysis=False, 
        show_plots=True, 
        feature_names=None
    ):
    """
    A pipeline function to perform clustering, visualize results, and analyze features.

    Parameters:
    - X (ndarray): The dataset (scaled or unscaled).
    - model_name (str): The clustering model to use (e.g., "KMeans").
    - n_clusters (int): Number of clusters for the model.
    - sample_size (int): Optional sample size for faster processing.
    - max_k (int): Maximum number of clusters for Elbow Method analysis.
    - elbow_analysis (bool): Whether to perform Elbow Method analysis.
    - show_plots (bool): Whether to display plots.
    - feature_names (list): Optional list of feature names.

    Returns:
    - feature_means (DataFrame): Mean values of features for each cluster.
    """
    print("\n--- Starting Clustering Pipeline ---")

    # Step 1: Preprocess the data
    print("\n--- Preprocessing Data ---")
    scaler = StandardScaler()
    preprocessor = Pipeline([('scaler', scaler)])
    X_processed = preprocessor.fit_transform(X)
    print("Data preprocessing complete.")

    # Step 2: Perform Elbow Method analysis (optional)
    if elbow_analysis:
        print("\n--- Performing Elbow Method Analysis ---")
        find_best_k_elbow(
            model_name=model_name, 
            X=X, 
            max_k=max_k, 
            sample_size=sample_size, 
            show=show_plots
        )

    # Step 3: Fit the selected clustering model
    print(f"\n--- Fitting {model_name} Model ---")
    model_results = fit_model(X=X, n_clusters=n_clusters, sample_size=sample_size, verbose=1)

    # Step 4: Visualize clustering results
    print("\n--- Visualizing Clustering Results ---")
    visualize_clustering_results(
        model_name=model_name, 
        data_for_viz=model_results['data'], 
        labels_for_viz=model_results['labels'], 
        show=show_plots
    )

    # Step 5: Analyze cluster features
    print("\n--- Analyzing Cluster Features ---")
    feature_means = analyze_cluster_features(
        model_name=model_name,
        X=model_results['data'], 
        labels=model_results['labels'], 
        feature_names=feature_names, 
        show=show_plots
    )

    print("\n--- Clustering Pipeline Complete ---")
    return feature_means










# ---     PIPELINE STARTS     ---

# --- Data Loading/Generation ---
X, _ = make_blobs(
    n_samples=100_000,
    n_features=20,
    centers=5,
    cluster_std=6,
    random_state=42
)

print("Data generation complete.")
# Run the clustering pipeline
feature_means = clustering_pipeline(
    X=X,
    model_name="KMeans",
    n_clusters=4,
    sample_size=10_000, # 100_000 = 100 seconds per training
    max_k=10,
    elbow_analysis=True,
    show_plots=True,
    feature_names=[f"Feature_{i+1}" for i in range(20)]
)

# Print the feature means
# print("\nFeature Means per Cluster:")
# print(feature_means)