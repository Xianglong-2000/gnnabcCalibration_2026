


[growth_sim_simple_experiment](https://github.com/Xianglong-2000/gnnabcCalibration_2026/blob/main/others/additional_analysis/growth_sim_simple_experiment.ipynb) was used to check the basic growth behaviors in our simulation modules. From the previous observations, we realized all simulated samples are growing a bit faster than the experimental samples, so we used this notebook to figure out the reasons. We eventually found that the targetVol parameter values don't match their associated startVol values, so part of the divisions would appear a bit earlier than they should.
