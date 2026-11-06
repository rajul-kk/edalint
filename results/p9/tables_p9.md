# P9 verdicts (mechanical, from results/p9/*.json)

- LK1 N14 ir_drop: not found (lower bound) (0 confirmed of 3544 test rows, 0 flagged)
- LK2 N14 ir_drop Vortex-large: proximity material False, severity None (S_same 0.20353242754936218, S_other 0.3069, q25 0.7441, share 0.013)
- LK2 N14 ir_drop openc910-1: proximity material False, severity None (S_same None, S_other 0.3416, q25 0.7441, share 0.016)
- LK2 N14 ir_drop zero-riscy: proximity material False, severity None (S_same 0.5853140950202942, S_other 0.2522, q25 0.7441, share 0.971)
- LK1 N14 congestion: not found (lower bound) (0 confirmed of 3544 test rows, 0 flagged)
- LK2 N14 congestion Vortex-large: proximity material False, severity None (S_same 0.25137779116630554, S_other 0.3767, q25 0.8719, share 0.013)
- LK2 N14 congestion openc910-1: proximity material False, severity None (S_same None, S_other 0.3787, q25 0.8719, share 0.016)
- LK2 N14 congestion zero-riscy: proximity material False, severity None (S_same 0.701504647731781, S_other 0.3207, q25 0.8719, share 0.971)
- LK1 N14 drc: not found (lower bound) (0 confirmed of 3543 test rows, 0 flagged)
- LK2 N14 drc Vortex-large: proximity material False, severity None (S_same 0.15167948603630066, S_other 0.2051, q25 0.4969, share 0.013)
- LK2 N14 drc openc910-1: proximity material False, severity None (S_same None, S_other 0.2008, q25 0.4969, share 0.016)
- LK2 N14 drc zero-riscy: proximity material False, severity None (S_same 0.39819103479385376, S_other 0.1972, q25 0.4969, share 0.971)
- LK1 ICCAD'23: not found (lower bound) (0 flagged)
- LK3 circuitnet_official_N14: none []
- LK3 circuitnet_official_N28: none []
- LK3 cldroute_N14: none []
- LK3 cldroute_N28: none []

| target | model | U1 abs/norm | U2 | U3 | U4 | T_id cov | per-design coverage C_id / C_xd |
|---|---|---|---|---|---|---|---|
| n14_ir_drop | ridge | True/True | False | False | True | 0.920 | zero-riscy 0.928/0.803; Vortex-large 0.739/0.539; openc910-1 0.900/0.726 |
| n14_ir_drop | gbm | True/True | False | False | True | 0.914 | zero-riscy 0.904/0.884; Vortex-large 0.800/0.745; openc910-1 0.945/0.931 |
| n14_congestion | ridge | True/True | False | False | True | 0.888 | zero-riscy 0.965/0.970; Vortex-large 0.786/0.795; openc910-1 0.935/0.939 |
| n14_congestion | gbm | True/True | False | False | True | 0.914 | zero-riscy 0.988/0.985; Vortex-large 0.866/0.858; openc910-1 0.983/0.980 |
| iccad23 | ridge | False/True | True | True | True | 0.957 | hidden 0.644/0.963 |
| iccad23 | gbm | False/True | True | True | True | 0.955 | hidden 0.734/0.949 |
