# Flagpol
Energy and polarization based on-line interference mitigation in radio astronomy. Methods are described in [this paper](https://arxiv.org/abs/2412.14775).

## Flagpol
Python code for on-line RFI mitigation using spectral kurtosis and polarization alignment of radio interferometric data.

Example:

```
  flagpol.py --MS data.MS --finite_prec --time_window_size 10 --freq_window_size 2
```

will use ```data.MS``` as the input data and perform flagging. Using ```--finite_prec``` will turn on finite precision *emulation*, hence slower, so just to test the flagging algorithms themselves, do not enable this option.

## RL
Python code for training a reinforcement learning (RL) agent for optimizing precision of arithmetic operations used in the flagging algorithm.

Example:

```
  main_sac.py --episodes 100000 --seed 3333
```
will train an RL model to optimize the precision of the computing routines (cuda, 32 bit or 16 bit) using the soft actor-critic algorithm. After an ensemble of such models are trained (with different random ```--seed```), you can store each model in directorites like ```mydir/run1/```, ```mydir/run2/```, ```mydir/run3/``` and so on. Thereafter, run the ensemble evaluation as

```
  eval_model.py --episodes 100000 --steps 100 --models 4 --path mydir
```

## Simul
Python code for simulating realistic data with known RFI, and performing RFI mitigation. Thereafter, calculating false alarm and missed detection probabilities.

## Requirements
pytorch, numpy, scipy, python-casacore, gymnasium, matplotlib

wo  2 apr 2025 10:20:45 CEST
