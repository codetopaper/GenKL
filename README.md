# GenKL: An iterative framework for resolving label ambiguity and label non-conformity in web images via a new Generalized KL divergence

The GenKL code for Clothing1M, Food101/Food101N, and mini WebVision 1.0 datasets are available here.

Some D_pre data for datasets Clothing1M, Food101/Food101N and mini WebVision 1.0 is provided in [Google drive](https://drive.google.com/drive/folders/1dP4m61BTNWMN-9vVJIqWZvmsoTWD3Syd?usp=sharing). 

To run the code for Clothing1M in the first iteration (for 3 trials with different random seeds):

##Trial 1
```
python train_clothing1m.py --folder_log iter0 --seed 0 --idv_prediction_vector clothing1m/idv_prediction_vectors/1/noisy1m.npy clothing1m/idv_prediction_vectors/2/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy
```
##Trial 2
```
python train_clothing1m.py --folder_log iter0 --seed 1 --idv_prediction_vector clothing1m/idv_prediction_vectors/1/noisy1m.npy clothing1m/idv_prediction_vectors/2/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy
```
##Trial 3
```
python train_clothing1m.py --folder_log iter0 --seed 2 --idv_prediction_vector clothing1m/idv_prediction_vectors/1/noisy1m.npy clothing1m/idv_prediction_vectors/2/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy
```

    
NC instance identification task: The 200 manually verified NC instances from the Clothing1M dataset can be found in NC_instances.txt.
