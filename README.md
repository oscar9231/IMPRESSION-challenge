# IMPRESSION-challenge

The feature processing consists of 2 steps: resampling and extraction.

1. resample.py

For the sample numbers that can be divided by integer, decimate function is applied on the features.

For the sample number that cannot, decimate function is first applied to downsample the features so that the number of features approximates the number of labels. Then FFT transformation is applied to downsample the features.

*exception*:
For participant's eye features, the first row is first removed then decimate function is applied, since the first row is basically 0. By removing one row, the sample number is the same as that in the user manual and can be divided by integer.

2. extract_feats.py

participant features, stimuli features, and labels are extracted respectively.
stimuli features have 44923 samples; participant features and labels both have 44923 * 40 samples.

For stimuli features (sti.csv):

column 0-16: audio features
column 17-34: facial features
column 35-374: eye features
column 375-end: unknown. no explanation, just said params and pose

For participant features (0.csv - 39.csv)
column 0-24: eye features
column 25-59: facial features
column 60-end: physiological features
