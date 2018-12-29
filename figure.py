from utils.config import DATA_DIR, PLOT_DIR
from utils.plot import *
import pickle

METHOD_COLOR = {
 "Prediction_1" : "firebrick",
 "Prediction_2" : "mediumblue",
}
completion_range = np.arange(0, 105, 5)

####################
## COMPARE METHOD ##
####################

#Caltrain
mean_dist_per_dr = pickle.load(open(DATA_DIR + "mean_dist_per_dr_Caltrain.pkl", "rb"))
nbc=25

data_plot = create_data_dic_plot_compare_method_nb_cluster(mean_dist_per_dr, completion_range)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
labels = ["Prediction_1", "Prediction_2"]
title = "a - San Francisco, Caltrain Station \n 25 Clusters"
xlabel = "Trajectory Completion, p (%)"
ylabel = r"$Q_{pred}(p)$ (km)"
plot_prediction_final_destination_compare_method(fig, ax, data_plot, labels, nbc, completion_range,
                                                 METHOD_COLOR,
                                                 "nb_cluster", title, xlabel, ylabel, loc=1, ylim=[300, 2350])
plt.savefig(PLOT_DIR + "compare_method_caltrain.png", dpi=100, bbox_inches="tight")
plt.close()


#Sao Bento
mean_dist_per_dr = pickle.load(open(DATA_DIR + "mean_dist_per_dr_SaoBento.pkl", "rb"))
nbc=45

data_plot = create_data_dic_plot_compare_method_nb_cluster(mean_dist_per_dr, completion_range)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
labels = ["Prediction_1", "Prediction_2"]
title = "b - Porto, Sao Bento Station \n 45 Clusters"
xlabel = "Trajectory Completion, p (%)"
ylabel = r"$Q_{pred}(p)$ (km)"
plot_prediction_final_destination_compare_method(fig, ax, data_plot, labels, nbc, completion_range,
                                                 METHOD_COLOR,
                                                 "nb_cluster", title, xlabel, ylabel, loc=1, ylim=[300, 2350])
plt.savefig(PLOT_DIR + "compare_method_saobento.png", dpi=100, bbox_inches="tight")
plt.close()