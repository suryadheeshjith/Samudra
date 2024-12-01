# Experiment inputs and outputs
DEPTH_LEVELS = ['2_5',
 '10_0',
 '22_5',
 '40_0',
 '65_0',
 '105_0',
 '165_0',
 '250_0',
 '375_0',
 '550_0',
 '775_0',
 '1050_0',
 '1400_0',
 '1850_0',
 '2400_0',
 '3100_0',
 '4000_0',
 '5000_0',
 '6000_0']

INPT_VARS = {
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_noFast_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"]
}
EXTRA_VARS = {
    "3D_all_hfds_anom": ["tauuo", "tauvo", "hfds", "hfds_anomalies"]
}
OUT_VARS = {
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_noFast_all": [
        k + str(j)
        for k in ["thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"]
}