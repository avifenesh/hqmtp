# MTP acceptance: ceiling (bf16 full-prec) vs quant (NVFP4)  [median of N runs]
# ceiling=research/mtp-heal/g7e/out-bf16.jsonl  quant=research/mtp-heal/g7e/out-nvfp4.jsonl
prompt          K    bf16   nvfp4   delta   hit% consist
------------------------------------------------------------
p1              1   0.841   0.868  -0.027   -3.2 PASS
p1              2   0.796   0.790  +0.006   +0.8 PASS
p1              3   0.683   0.674  +0.009   +1.3 PASS
p1              4   0.512   0.583  -0.071  -13.9 PASS
p2              1   0.753   0.778  -0.025   -3.3 PASS
p2              2   0.585   0.623  -0.038   -6.5 PASS
p2              3   0.455   0.503  -0.048  -10.5 PASS
p2              4   0.387   0.446  -0.059  -15.2 PASS
p3              1   0.969   0.829  +0.140  +14.4 PASS
p3              2   0.911   0.740  +0.171  +18.8 PASS
p3              3   0.615   0.651  -0.036   -5.9 PASS
p3              4   0.461   0.571  -0.110  -23.9 PASS
p4              1   0.939   0.789  +0.150  +16.0 PASS
p4              2   0.902   0.698  +0.204  +22.6 PASS
p4              3   0.870   0.587  +0.283  +32.5 PASS
p4              4   0.782   0.544  +0.238  +30.4 PASS
p5              1   0.910   0.753  +0.157  +17.3 PASS
p5              2   0.880   0.655  +0.225  +25.6 PASS
p5              3   0.829   0.513  +0.316  +38.1 PASS
p5              4   0.798   0.456  +0.342  +42.9 PASS
p6              1   0.910   0.829  +0.081   +8.9 PASS
p6              2   0.862   0.673  +0.189  +21.9 PASS
p6              3   0.807   0.567  +0.240  +29.7 PASS
p6              4   0.699   0.472  +0.227  +32.5 PASS
agentloop-t1    1   0.947   0.903  +0.044   +4.6 PASS
agentloop-t2    1   0.984   0.947  +0.037   +3.8 PASS
agentloop-t3    1   1.000   0.969  +0.031   +3.1 PASS
agentloop-t4    1   0.992   0.962  +0.030   +3.0 PASS
agentloop-t5    1     -     0.896     -      -   PASS
agentloop-t6    1   0.992   0.848  +0.144  +14.5 PASS
agentloop-t7    1   0.992   0.984  +0.008   +0.8 PASS
agentloop-t8    1   1.000   0.925  +0.075   +7.5 PASS
