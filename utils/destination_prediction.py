import numpy as np
import pandas as pd
import warnings
from traj_dist.cydist.basic_geographical import c_great_circle_distance

warnings.filterwarnings("ignore")


def create_cumsum_data(data):
    """
    USAGE
    Compute cumsum for each "score column" of data according to id_traj column.

    INPUT
    data : pandas DataFrame

    OUTPUT
    data_cumsum : pandas DataFrame where each score columns represent cumulative sum of the different score
    according to id_traj columns
    """
    scores_columns = filter(lambda c: c.startswith("score_"), data.columns)
    data_cumsum = data.groupby("id_traj")[scores_columns].apply(lambda t: t.cumsum())
    return data_cumsum


def create_mean_dropoff_per_cv(data_all, cv_list):
    nb_cv = len(cv_list)
    mean_dropoff_per_cv = dict([(cv,{}) for cv in range(nb_cv)])
    median_dropoff_per_cv = dict([(cv,{}) for cv in range(nb_cv)])
    for icv,cv in enumerate(cv_list):
        data_cv_train = data_all[np.logical_not(data_all.id_traj.isin(cv))]
        data_cvtcid = data_cv_train.groupby(["traj_clust","id_traj"]).last()[["lons","lats"]]

        data_cv_means = data_cvtcid.mean(level="traj_clust")
        mean_dropoff_cv = [(tc,[df.lons,df.lats]) for tc,df in data_cv_means.iterrows()]
        mean_dropoff_per_cv[icv].update(mean_dropoff_cv)

        data_cv_medians = data_cvtcid.median(level="traj_clust")
        median_dropoff_cv = [(tc,[df.lons,df.lats]) for tc,df in data_cv_medians.iterrows()]
        median_dropoff_per_cv[icv].update(median_dropoff_cv)
    return mean_dropoff_per_cv, median_dropoff_per_cv


def create_data_exp_norm(data, scores_columns):
    data_cv_score_exp = np.exp(data[scores_columns])
    inf_loc = data_cv_score_exp==np.inf
    data_cv_score_exp_norm = data_cv_score_exp.divide(data_cv_score_exp.sum(1),0)
    data_cv_score_exp_norm[inf_loc] = 1.0
    data_cv_score_exp_norm = data_cv_score_exp_norm.divide(data_cv_score_exp_norm.sum(1),0)
    return data_cv_score_exp_norm



def create_data_pct_score(data, cv_list):
    scores_columns = filter(lambda c: c.startswith("score_"), data.columns)
    data_score_norm = []
    for icv in range(len(cv_list)):
        data_cv_test = data[data.id_traj.isin(cv_list[icv])]
        data_cv_score_exp_norm = create_data_exp_norm(data_cv_test, scores_columns)
        data_score_norm.append(data_cv_score_exp_norm)
    data_pct_score_df = pd.concat(data_score_norm).reindex(data.index)
    return data_pct_score_df



def create_dropoff_dic_per_traj(data):
    dropoff_dic = dict([(id,[df.lons,df.lats]) for id,df in data.groupby("id_traj")[["lons","lats"]].last(
            ).iterrows()])
    return dropoff_dic


def create_pred_columns(data,scores_columns):
    pred_columns = data[scores_columns].idxmax(1).apply(lambda t:t.split(
                "_")[-1])
    return pred_columns


def row_pred_dist_1(row, dropoff_dic, dics):
    coord_dropoff_row =  dropoff_dic[row[0]]
    lons_row = coord_dropoff_row[0]
    lats_row = coord_dropoff_row[1]
    pred = row[1]
    lons_pred = dics[pred][0]
    lats_pred = dics[pred][1]
    dist = c_great_circle_distance(lons_row, lats_row, lons_pred, lats_pred)
    return dist

def create_pred_dist_columns_1(data, dropoff_dic, mean_dropoff_cv):
    ds = data[["id_traj", "pred"]].values
    pred_dist_columns_1 = map(lambda d : row_pred_dist_1(d,dropoff_dic,mean_dropoff_cv),ds)
    return pred_dist_columns_1


def row_pred_dist_2_(row, dropoff_dic):
    coord_dropoff_row =  dropoff_dic[row[2]]
    lons_row = coord_dropoff_row[0]
    lats_row = coord_dropoff_row[1]
    lons_pred = row[0]
    lats_pred = row[1]
    dist = c_great_circle_distance(lons_row, lats_row, lons_pred, lats_pred)
    return dist

def create_pred_dist_columns_2(data, dropoff_dic, mean_dropoff_cv,scores_columns):
    lons_line = [mean_dropoff_cv[int(c.split("_")[-1])][0] if int(c.split("_")[-1]) in
                                                 mean_dropoff_cv.keys() else 0   for c in  scores_columns]
    lons_lines = [lons_line for _ in range(len(data))]
    lons_dest  = pd.DataFrame(lons_lines,index=data.index, columns=scores_columns)


    lats_line = [mean_dropoff_cv[int(c.split("_")[-1])][1] if int(c.split("_")[-1]) in
                                                 mean_dropoff_cv.keys() else 0   for c in scores_columns]
    lats_lines = [lats_line for _ in range(len(data))]
    lats_dest  = pd.DataFrame(lats_lines,index=data.index, columns=scores_columns)

    lons_dest_pct = (lons_dest.multiply(data[scores_columns])).sum(1)
    lats_dest_pct = (lats_dest.multiply(data[scores_columns])).sum(1)

    pred_dist_columns_2 = map(lambda d : row_pred_dist_2_(d,dropoff_dic),zip(lons_dest_pct,lats_dest_pct,
                                                                            data["id_traj"]))

    return pred_dist_columns_2



def create_data_preds(data, mean_dropoff_per_cv, cv_list):

    data_preds = []
    scores_columns = filter(lambda c: c.startswith("score_"), data.columns)
    for icv in range(len(cv_list)):
        data_cv_test = data[data.id_traj.isin(cv_list[icv])]
        dropoff_dic = create_dropoff_dic_per_traj(data_cv_test)


        data_cv_test.loc[:,"pred"] = create_pred_columns(data_cv_test,scores_columns).astype(int)
        data_cv_test["dist_1"] = create_pred_dist_columns_1(data_cv_test, dropoff_dic, mean_dropoff_per_cv[icv])
        data_cv_test["dist_2"] = create_pred_dist_columns_2(data_cv_test, dropoff_dic, mean_dropoff_per_cv[icv], scores_columns)
        data_preds.append(data_cv_test)
    data_preds = pd.concat(data_preds).reindex(data.index)
    return data_preds


def build_exogenous_data(data):

    #dist update
    data_coord = data[["lons", "lats", "id_traj"]].values
    dist = map(lambda x,y : 0 if x[2] != y[2] else c_great_circle_distance(x[0], x[1], y[0], y[1]),
               data_coord[:-1],data_coord[1:])
    data.loc[:,"dist"] = np.hstack((0, dist))
    data.loc[:,"rdist"] = data.groupby("id_traj")['dist'].apply(lambda x: x.cumsum())
    length_trip_per_traj = dict(data.groupby("id_traj")["rdist"].last())
    data.loc[:,"rdist_pct"] = data.rdist.values/np.array([length_trip_per_traj[x] for x in data.id_traj],dtype=float)


def create_mean_dist_per_dr(data_all, completion_range):
    dist_columns = filter(lambda c: c.startswith("dist"), data_all.columns)
    mean_dist_per_dr =  {}
    for comp in completion_range:
        data_comp = data_all[data_all.rdist_pct<=comp]
        data_gbit_mean = data_comp.groupby("id_traj").last().mean()
        mean_dist_per_dr.update({int(comp*100) : dict(data_gbit_mean[dist_columns]) })
        mean_dist_per_dr[int(comp*100)].update({"rdist_pct_mean" : data_gbit_mean["rdist_pct"]*100})
    return mean_dist_per_dr