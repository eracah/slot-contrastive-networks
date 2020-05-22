from scripts.table_utils import get_df

if __name__ == "__main__":
    acc_methods = ['random-cnn',
                   'slot-stdim_scn_sdl',
                   'cswm',
                   'stdim',
                   'supervised']
    disent_methods = ['random-cnn',
                      'slot-stdim_scn_sdl',
                      "cswm",
                      'supervised']
    ablations = ['slot-stdim_scn', 'slot-stdim_scn_sdl']

    map_dict = {"slot-stdim_hcn": "slot-dim_loss1only", "slot-stdim_scn": "scn_loss1only", "slot-stdim_scn_sdl": "scn",
                "slot-stdim_hcn_smdl": "slot-stdim"}


    metric_name = 'concat_overall_localization_avg_r2_lin_reg'
    final_df, count_df, mean_df,stderr_df = get_df(metric_name=metric_name, methods=acc_methods)
    dcid_df, count_df, dcid_mean_df, dcid_std_df = get_df(metric_name="dci_d_gbt", methods=disent_methods)
    dcic_df, count_df, dcic_mean_df, dcic_std_df = get_df(metric_name="dci_c_gbt", methods=disent_methods)
