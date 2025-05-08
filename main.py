import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AffinityPropagation, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA 


# --- Configuration ---
N_SAMPLES = 1_000_000 # Target number of samples
N_FEATURES = 20
N_CLUSTERS = 5 
SAMPLE_SIZE_FOR_EXPENSIVE_ALGORITHMS = 1000 # Reduce sample size for slow algorithms
SAMPLE_SIZE_FOR_METRICS = 10000 # Reduce sample size for metric calculation if needed

OUTPUT_FILES = "output_files" # Directory to save output files (ensure it exists)
if not os.path.exists(OUTPUT_FILES):
    os.makedirs(OUTPUT_FILES)
    print(f"Created directory: {OUTPUT_FILES}")

# --- Data Loading/Generation ---
# In a real scenario, load your data from the database here
# Example: data = pd.read_sql("SELECT * FROM your_table", your_db_connection)
# For demonstration, we generate synthetic data:
print(f"Generating synthetic data ({N_SAMPLES} samples, {N_FEATURES} features)...")
X = np.random.rand(N_SAMPLES, N_FEATURES)
# Convert to DataFrame if needed (sklearn works well with numpy arrays)
# data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(N_FEATURES)])
print("Data generation complete.")

# --- Preprocessing ---
# Scaling is crucial for distance-based algorithms
scaler = StandardScaler()

# Optional: PCA for dimensionality reduction (can speed up some algorithms)
# pca = PCA(n_components=0.95) # Retain 95% of variance or specify n_components
# preprocessor = Pipeline([('scaler', scaler), ('pca', pca)])
preprocessor = Pipeline([('scaler', scaler)])

print("Preprocessing data...")
start_time = time.time()
X_processed = preprocessor.fit_transform(X)
# If using sampling for expensive algorithms, create the sample here
if SAMPLE_SIZE_FOR_EXPENSIVE_ALGORITHMS < N_SAMPLES:
    sample_indices = np.random.choice(X_processed.shape[0], SAMPLE_SIZE_FOR_EXPENSIVE_ALGORITHMS, replace=False)
    X_processed_sample = X_processed[sample_indices, :]
else:
    X_processed_sample = X_processed # Use full data if sample size is not smaller

# Sample for metrics calculation
if SAMPLE_SIZE_FOR_METRICS < N_SAMPLES:
     metric_indices = np.random.choice(X_processed.shape[0], SAMPLE_SIZE_FOR_METRICS, replace=False)
     X_processed_metrics_sample = X_processed[metric_indices, :]
else:
    X_processed_metrics_sample = X_processed

print(f"Preprocessing complete. Time taken: {time.time() - start_time:.2f} seconds")


# --- Model Definitions ---
# Note: Parameters often need tuning based on the dataset
models = {
    # Use MiniBatchKMeans for large datasets
    "MiniBatchKMeans": MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=1024, n_init=3, random_state=42),
    # Adding KMeans - will run on sample due to potential slowness on full large dataset
    "KMeans": KMeans(n_clusters=N_CLUSTERS, n_init=3, random_state=42),
    # # DBSCAN can be slow (O(n^2) worst case). Consider parameters carefully or use HDBSCAN library.
    # # Might require running on the smaller sample (X_processed_sample)
    # "DBSCAN": DBSCAN(eps=0.5, min_samples=5, n_jobs=-1), # eps needs tuning!
    # # AgglomerativeClustering is typically O(n^2) - very slow for 1M points. Run on sample.
    # "AgglomerativeClustering": AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward'), # Run on sample
    # # GMM can be slow, depends on components and iterations
    # "GaussianMixture": GaussianMixture(n_components=N_CLUSTERS, random_state=42),
    # # AffinityPropagation is O(n^2) - very slow. Run on sample.
    # "AffinityPropagation": AffinityPropagation(random_state=42), # Run on sample
    # # MeanShift can be slow. Run on sample. Bandwidth selection is crucial.
    # "MeanShift": MeanShift(n_jobs=-1), # Run on sample
}

results = {}

# --- Pipeline Execution ---
for name, model in models.items():
    print(f"\n--- Running {name} ---")
    start_time = time.time()

    # Decide whether to run on full data or sample based on algorithm complexity
    if name in ["DBSCAN", "AgglomerativeClustering", "AffinityPropagation", "MeanShift"]: #"KMeans", 
        print(f"Running {name} on a sample of size {X_processed_sample.shape[0]} due to complexity.")
        data_to_fit = X_processed_sample
        data_for_metrics = data_to_fit # Evaluate on the same sample
    else:
        print(f"Running {name} on the full dataset.")
        data_to_fit = X_processed
        data_for_metrics = X_processed_metrics_sample # Use metric sample

    try:
        # Fit the model
        model.fit(data_to_fit)

        # Get labels
        if hasattr(model, "labels_"):
            labels = model.labels_
        else:
            labels = model.predict(data_to_fit) # For GMM

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Fitting complete. Time taken: {elapsed_time:.2f} seconds")

        # --- Evaluation (on a sample for efficiency) ---
        # Avoid calculating metrics if all points are noise (-1) or only one cluster found
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0) # Exclude noise label if present

        if n_clusters > 1:
            print(f"Calculating metrics on sample size {data_for_metrics.shape[0]}...")
            metric_start_time = time.time()
            # Predict labels for the metric sample if the model was trained on a different sample
            if data_to_fit is not data_for_metrics:
                 if hasattr(model, "predict"):
                     metric_labels = model.predict(data_for_metrics)
                 else: # For algorithms like DBSCAN fitted on sample, predicting on new data isn't straightforward
                     print("Skipping metrics: Model fitted on sample doesn't directly predict on new data points easily.")
                     silhouette = np.nan
                     davies_bouldin = np.nan
                 # Ensure metric_labels is defined before use
                 if 'metric_labels' in locals():
                     metric_unique_labels = np.unique(metric_labels)
                     metric_n_clusters = len(metric_unique_labels) - (1 if -1 in metric_unique_labels else 0)
                     if metric_n_clusters > 1:
                        silhouette = silhouette_score(data_for_metrics, metric_labels)
                        davies_bouldin = davies_bouldin_score(data_for_metrics, metric_labels)
                     else:
                        print(f"Skipping metrics: Not enough clusters ({metric_n_clusters}) found in metric sample.")
                        silhouette = np.nan
                        davies_bouldin = np.nan
                 else: # metric_labels was not assigned (e.g. DBSCAN case)
                    silhouette = np.nan
                    davies_bouldin = np.nan


            else: # Model trained on the same data used for metrics (or full data)
                silhouette = silhouette_score(data_for_metrics, labels[metric_indices] if data_to_fit is X_processed else labels)
                davies_bouldin = davies_bouldin_score(data_for_metrics, labels[metric_indices] if data_to_fit is X_processed else labels)

            print(f"Metrics calculation time: {time.time() - metric_start_time:.2f} seconds")
            print(f"Number of clusters found: {n_clusters}")
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
            results[name] = {
                'time': elapsed_time,
                'n_clusters': n_clusters,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'labels': labels if data_to_fit is X_processed_sample else 'Stored separately or re-predict',
                'error': np.nan  # Add error key with NaN for successful runs
            }
        else:
            print(f"Skipping metrics: Only {n_clusters} cluster(s) found.")
            results[name] = {
                'time': elapsed_time,
                'n_clusters': n_clusters,
                'silhouette': np.nan,
                'davies_bouldin': np.nan,
                'labels': labels if data_to_fit is X_processed_sample else 'Stored separately or re-predict',
                'error': np.nan  # Add error key with NaN
            }

    except Exception as e:
        print(f"Failed to run {name}: {e}")
        results[name] = {
            'time': time.time() - start_time,
            'n_clusters': np.nan,  # Add metric keys with NaN for failed runs
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'labels': 'N/A due to error',
            'error': str(e)
        }


# --- Summary ---
print("\n--- Pipeline Summary ---")
summary_df = pd.DataFrame(results).T
print(summary_df[['time', 'n_clusters', 'silhouette', 'davies_bouldin', 'error']])

# --- Visualization Example (using PCA and MiniBatchKMeans results) ---
selected_model = "KMeans"
def fit_selected_model(model_name, X):
    scaler = StandardScaler()
    preprocessor = Pipeline([('scaler', scaler)])
    start_time = time.time()
    X_processed = preprocessor.fit_transform(X)
    model.fit(X_processed)
    labels = model.predict(X_processed)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Fitting complete. Time taken: {elapsed_time:.2f} seconds")
    return labels, X_processed



def visualize_clustering_results(
        model_name, 
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
    # Use a colormap that handles many clusters
    unique_labels_viz = np.unique(labels_for_viz)
    colors = cm.get_cmap('viridis', len(unique_labels_viz))

    for i, label in enumerate(unique_labels_viz):
        cluster_color = 'gray' if label == -1 else colors(i)
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
        model_name, 
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
    
    if show:
        plt.show()
    
    # Save the figure
    output_file_name = f"{OUTPUT_FILES}/feature_heatmap_{model_name}.png"
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_file_name}")

    return cluster_feature_means

# Example usage
labels, data = fit_selected_model(selected_model, X)
visualize_clustering_results(
    selected_model, 
    data, 
    labels, 
    show=True
)
feature_means = analyze_cluster_features(
    selected_model, 
    data, 
    labels, 
    show=True
)

print("\nPipeline finished.")