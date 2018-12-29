import pickle
from utils.config import DATA_DIR, DATA_CLUSTERING_DIR
from utils.destination_prediction import *

## Caltrain
print("Caltrain Dataset")

print("Download clustering results")
# Download cv_list (for result reproduction)
output_cv = DATA_CLUSTERING_DIR + "cv_list_Caltrain.pkl"
cv_list = pickle.load(open(output_cv, "r"))

# Download data
data_original = pd.read_pickle(DATA_CLUSTERING_DIR + "Caltrain.pkl")[["id_traj", "lons", "lats"]].reset_index(drop=True)
nb_traj = len(data_original.id_traj.unique())


# Download trajectory's cluster labels
labels = pickle.load(open(DATA_CLUSTERING_DIR+"caltrain_traj_labels.pkl", "rb"))
nb_traj_class = labels.max()+1
data_original["traj_clust"] = [labels[idt] for idt in data_original.id_traj]


print("Compute GMM scores")
# Download GMM scores and compute cumulate score of trajectories
data_score = pd.read_pickle(DATA_CLUSTERING_DIR+"gmm_scores_Caltrain.pkl")
data_cumsum = create_cumsum_data(data_score)

# Create pct score
data_pct_score = create_data_pct_score(data_original.join(data_cumsum), cv_list)
data = data_original.join(data_pct_score)

print("Compute mean destination per cluster trajectories")
# Create Mean dropoff per cv
mean_dropoff_per_cv, median_dropoff_per_cv = create_mean_dropoff_per_cv(data_original, cv_list)

print("Compute destination prediction and error distance to true destination")
# Compute distance
data_preds = create_data_preds(data, mean_dropoff_per_cv, cv_list)
build_exogenous_data(data_preds)

print("Compute error distance per completion range and save results")
# Compute distance per completion_range
completion_range = np.arange(0, 1.05, 0.05)
mean_dist_per_dr = create_mean_dist_per_dr(data_preds, completion_range)
pickle.dump(mean_dist_per_dr, open(DATA_DIR + "mean_dist_per_dr_Caltrain.pkl", "wb"))




## Caltrain
print("Sao Bento Dataset")

print("Download clustering results")
# Download cv_list (for result reproduction)
output_cv = DATA_CLUSTERING_DIR + "cv_list_SaoBento.pkl"
cv_list = pickle.load(open(output_cv, "r"))

# Download data
data_original = pd.read_pickle(DATA_CLUSTERING_DIR + "Sao_bento.pkl")[["id_traj", "lons", "lats"]].reset_index(drop=True)
nb_traj = len(data_original.id_traj.unique())


# Download trajectory's cluster labels
labels = pickle.load(open(DATA_CLUSTERING_DIR+"sao_bento_traj_labels.pkl", "rb"))
nb_traj_class = labels.max()+1
data_original["traj_clust"] = [labels[idt] for idt in data_original.id_traj]


print("Compute GMM scores")
# Download GMM scores and compute cumulate score of trajectories
data_score = pd.read_pickle(DATA_CLUSTERING_DIR+"gmm_scores_saobento.pkl")
data_cumsum = create_cumsum_data(data_score)

# Create pct score
data_pct_score = create_data_pct_score(data_original.join(data_cumsum), cv_list)
data = data_original.join(data_pct_score)

print("Compute mean destination per cluster trajectories")
# Create Mean dropoff per cv
mean_dropoff_per_cv, median_dropoff_per_cv = create_mean_dropoff_per_cv(data_original, cv_list)

print("Compute destination prediction and error distance to true destination")
# Compute distance
data_preds = create_data_preds(data, mean_dropoff_per_cv, cv_list)
build_exogenous_data(data_preds)

print("Compute error distance per completion range and save results")
# Compute distance per completion_range
completion_range = np.arange(0, 1.05, 0.05)
mean_dist_per_dr = create_mean_dist_per_dr(data_preds, completion_range)
pickle.dump(mean_dist_per_dr, open(DATA_DIR + "mean_dist_per_dr_SaoBento.pkl", "wb"))