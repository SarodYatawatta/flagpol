# Flagpol
Energy and polarization based on-line interference mitigation in radio astronomy.

## Flagpol
Python code for on-line RFI mitigation using spectral kurtosis and polarization alignment of radio interferometric data.

Example:

```
  flagpol.py --MS data.MS --finite_prec --time_window_size 10 --freq_window_size 2
```

will use ```data.MS``` as the input data and perform flagging.

## RL
Python code for training an RL agent for optimizing precision of arithmetic operations used in the flagging algorithm.

Example:

```
  main_sac.py --episodes 100000 --seed 3333
```
will train an RL model to optimize the precision of the computing routines using the soft actor-critic algorithm.

wo 18 dec 2024 15:16:16 CET
