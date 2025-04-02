This code simulates a data block, ~ 1 sec duration and 200 kHz bandwidth, sampled at 1e6 samples and 512 channels. All 4 XX,XY,YX,YY correlations are simulated. This data block consists of

  * sky noise (power law)
  * sky sources (discrete sources, at random locations)
and RFI
  * transient RFI, randomly determined parameters: time x frequency window, intensity, polarization level, modulating signal (fixed or sinusoidal)
  * coherent RFI, randomly determined parameters: bandwidth, tone frequency


Thereafter, the RFI mitigation is performed on this data block, using a window size of 10 x 2 time-frequency samples.

How to run the script 

```
 mitigator.py --runs 400 --data_time_window_size 2000 --data_freq_window_size 512 --time_window_size 20 --freq_window_size 2
```

The output will show the false alarm (Pfa) and missed detection (Pmiss) probabilities for various interference to noise ratios (INR).
