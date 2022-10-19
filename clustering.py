import mlflow
from serotiny.ml_ops.mlflow_utils import download_artifact
from cytodata_aics.vae_utils import get_ranked_dims

import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

run_name = "vae_3d_run_20221018_152122"
mlflow.set_tracking_uri("http://mlflow.mlflow.svc.cluster.local")

with download_artifact("dataframes/embeddings.csv", experiment_name="cytodata_chapter_vae", run_name=run_name) as path:
    embeddings = pd.read_csv(path)
    
with download_artifact("dataframes/stats_per_dim_test.csv", experiment_name="cytodata_chapter_vae", run_name=run_name) as path:
    kl_per_dimension = pd.read_csv(path)
    
ranked_z_dim_list, mu_std_list, mu_mean_list = get_ranked_dims(kl_per_dimension, 0, 8)
ranked_z_dim_list = [f"mu_{i}" for i in ranked_z_dim_list]
updated_ranks = [f"mu_{i+1}" for i in range(8)]

embeddings = embeddings[[i for i in embeddings.columns if i in ranked_z_dim_list] + ['split']]

rename_cols = {}
for i, j in zip(ranked_z_dim_list, updated_ranks):
    rename_cols[i] = j
embeddings.rename(columns = rename_cols, inplace=True)
embeddings = embeddings.reindex(sorted(embeddings.columns), axis=1)

train = embeddings[embeddings['split']=='train'].drop(columns=['split'])
test = embeddings[embeddings['split']!='train'].drop(columns=['split'])

reducer = umap.UMAP()
projection = reducer.fit(train)
kmeans = KMeans(n_clusters=5, random_state=0).fit(train)

projected = projection.transform(test)
clusters = kmeans.predict(test)
plt.scatter(
    projected[:, 0],
    projected[:, 1],
    c=clusters)
print("Silhouette:", silhouette_score(test, clusters))
plt.show()