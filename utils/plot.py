import numpy as np
import matplotlib.pyplot as plt

def create_data_dic_plot_compare_method_nb_cluster(data_dic_plot, completion_range):
    data_plot = []
    for comp in completion_range:
        data_plot.append([comp,data_dic_plot[comp]["dist_1"],data_dic_plot[comp]["dist_2"]])
    data_plot = np.array( sorted(data_plot,key=lambda j : j[0]) )
    return data_plot


def plot_prediction_final_destination_compare_method(fig, ax, data_plot, labels, plot_ind, x_range,color_dic,
                                                fix_param='',title=None,xlabel=None,
                                                ylabel="Mean error prediction \n to the destination (in Km)",loc=0,
                                                ylim="None" ):
        min_dist = np.inf
        max_dist = - np.inf
        for eni,label in enumerate(labels):
            ax.plot(data_plot[:,0], data_plot[:,eni+1], color=color_dic[label], label=label, marker=".",linewidth=3)
            min_dist = min(min_dist, np.nanmin(data_plot[:,eni+1]))
            max_dist = max(max_dist, np.nanmax(data_plot[:,eni+1]))
        min_dist = min_dist - min_dist % 50
        max_dist = (max_dist + 50) - (max_dist + 50) % 50
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        ax.set_ylabel(ylabel, fontsize=35)

        ax.set_xticks(x_range)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
            tick.label.set_rotation(45)

        ax.set_ylim(min_dist, max_dist)
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        if fix_param =="accomplishement":
            ax.set_xticklabels(map(lambda x : str(int(x)),x_range))
            if title is None:
                ax.set_title("Mean Error Prediction of VS Number of Cluster P  \n Percentage of accomplishment : %.2f"
                         %plot_ind, fontsize=32)
            else:
                ax.set_title(title, fontsize=45)
            if xlabel is None:
                ax.set_xlabel("Number of Cluster", fontsize=35)
            else:
                ax.set_xlabel(xlabel, fontsize=35)
        elif fix_param =="nb_cluster":
            xtl = [str(x) if x%10==0 else "" for x in x_range]
            ax.set_xticklabels(xtl)
            if title is None:
                ax.set_title("Mean Error Prediction of VS Number of Cluster P  \n Number of clusters : %d"
                         %plot_ind, fontsize=32)
            else:
                ax.set_title(title, fontsize=45)
            if xlabel is None:
                ax.set_xlabel("Trajectory Accomplishment", fontsize=35)
            else:
                ax.set_xlabel(xlabel, fontsize=35)
        else:
            raise ValueError
        if ylim!="None":
            ax.set_ylim(ylim[0],ylim[1])
            ax.set_yticks(np.arange(ylim[0], ylim[1]+50 ,50))
            ax.set_yticklabels([str(float(x)/1000) if x%200==0 else ""  for x in np.arange(ylim[0], ylim[1]+50 , 50)])
        else:
            ax.set_ylim(min_dist, max_dist)
            ax.set_yticks(np.arange(min_dist, max_dist+50 ,50))
            ax.set_yticklabels([str(float(x)/1000) for x in np.arange(min_dist, max_dist+50 , 50)])

        if not("Porto" in title):
            plt.legend(loc=loc, fontsize=25)