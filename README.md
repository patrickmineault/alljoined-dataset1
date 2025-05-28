# Alljoined Dataset

This repo includes all the scripts we used to create a stimulus paradigm, collect data with a bioesmi device, preprocess it, and upload to huggingface. We also did some analysis work in 2_analysis.

## Benchmark decoding script

A simple 1-nearest-neighbor benchmark is available in `2_analysis/nn_decode.py`. It assumes each `.h5` file in `final_hdf5/<FREQ>/` contains two repetitions per stimulus. Run it with:

```bash
python 2_analysis/nn_decode.py --dataset_dir /path/to/eeg_data --freq_band 05_125
```

The script compares the first and second repetitions for each stimulus within a session and reports the decoding accuracy.
