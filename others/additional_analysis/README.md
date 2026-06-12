This [additional_analysis](https://github.com/Xianglong-2000/gnnabcCalibration_2026/tree/main/others/additional_analysis) includes all the analysis work needed to make the research progress of this project. Please find the descriptions of each analysis notebook below:

[growth_sim_simple_experiment](https://github.com/Xianglong-2000/gnnabcCalibration_2026/blob/main/others/additional_analysis/growth_sim_simple_experiment.ipynb) was used to check the basic growth behaviors in our simulation modules. From the previous observations, we realized all simulated samples are growing a bit faster than the experimental samples, so we used this notebook to figure out the reasons. We eventually found that the targetVol parameter values don't match their associated startVol values, so part of the divisions would appear a bit earlier than they should. (updated on 2026/06/11)



[check_empty_features_after_fillna](https://github.com/Xianglong-2000/gnnabcCalibration_2026/blob/main/others/additional_analysis/check_empty_features_after_fillna.ipynb) was used to check empty features in the simulated data after the filling-in preprocessing. (updated on 2026/03/10)

[check_sim_param_dist](https://github.com/Xianglong-2000/gnnabcCalibration_2026/blob/main/others/additional_analysis/check_sim_param_dist.ipynb) was used to check the observable parameter distributions in our simulation module setup. We compared the simulated distributions to the experimental distributions, and see if there's any discrepancy. We eventually realized all observable parameter distribution setup in our simulation module should be okay. (updated on 2026/03/10)
