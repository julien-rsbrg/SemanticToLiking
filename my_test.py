import os
import numpy as np

import dash_src.display as display
import dash_src.utils as utils
import dash_src.data_load as data_load

CONFIG = {
    "src_results_path":"experiments_results/3-fold_cross_validation",
    "participant_data_path":os.path.join("data/processed","participant_data.csv")
}

id_to_models_info, all_studies_summaries = data_load.load_data(exp_path=CONFIG["src_results_path"], participant_data_path=CONFIG["participant_data_path"])


var = "train_BIC_0"
show_legend = ""
groups_kept = ["1NN_mean","GAT_liking_sim_amp_3NN_3ExpNN"]
join_on="participant_folder_name"
join_on_aggr="mean"
group_by="model_id"
data=all_studies_summaries

data, _ = utils.restrict_data(data,group_by,groups_kept,np.zeros(len(data),dtype=bool))

fig = display.create_grouped_heatmap(
            data=data,
            var=var,
            fn="mean difference y-x",
            join_on=join_on,
            group_by=group_by,
            join_on_aggr=join_on_aggr,
            showticks=show_legend == "show legend"
        )
fig.show()


def fun():
    """
    Parameters:
    -----------
    - OK
    
    """

    return 


fun()