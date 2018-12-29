# Destination Prediction of Trajectory  #

Python code to re-produce results and illustrations of *Destination Prediction by Trajectory Distribution
Based Model* detailed in publications [1] :

# Dataset

Two datasets are used in the publication :

* **Caltrain dataset** is composed of 4127 trajectories from taxis which begin their trip at Caltrain station, San Francisco.
It is a subset of the cabspotting data set [3].

* **Sao Bento dataset** is composed of 19423 trajectories from taxis which begin their trip at Sao Bento station, Porto.
It is a subset of train dataset of the Kaggle ECML/PKDD 15: Taxi Trajectory Prediction (I) competition [3].

# Trajectory Clustering and Classification


The different scripts to generate the two subsets described before and to produce the clustering of these trajectories,
can be found in the following repository : [https://github.com/bguillouet/trajectory_classification](https://github.com/bguillouet/trajectory_classification).

All the steps described in the [README.md](https://github.com/bguillouet/trajectory_classification/blob/master/README.md) 
of the *trajectory_classification* repository has to be executed before to run the script of this repository.

# Trajectory Prediction

 1. `utils/config.py`: Set the variable `DATA_CLUSTERING_DIR` to the path of the [data](https://github.com/bguillouet/trajectory_classification/tree/master/data) directory of the *trajectory_classification* repository.
 2. `generation_destination_prediction.py`: Run the final destination prediction method described in [1]. Save the necessary data in order to produce the figure.
 3. `figure.py`: Produce the following png file :
 
![Caltrain classification](https://raw.githubusercontent.com/bguillouet/trajectory_classification/master/plot/compare_method_caltrain.png)

![Sao Bento classification](https://raw.githubusercontent.com/bguillouet/trajectory_classification/master/plot/compare_method_saobento.png)

 



* [1] BESSE, Philippe C., GUILLOUET, Brendan, LOUBES, Jean-Michel, et al. Destination Prediction by Trajectory Distribution-Based Model. IEEE Transactions on Intelligent Transportation Systems, 2017.
