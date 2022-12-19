# GenKL: An iterative framework for resolving label ambiguity and label non-conformity in web images via a new Generalized KL divergence

The GenKL code for Clothing1M, Food101/Food101N, and mini WebVision 1.0 datasets are available here.

#### Note that some D_pre data for datasets Clothing1M, Food101/Food101N and mini WebVision 1.0 is provided in [Google drive](https://drive.google.com/drive/folders/1dP4m61BTNWMN-9vVJIqWZvmsoTWD3Syd?usp=sharing). 

#### To run the code for Clothing1M in the first iteration (for 3 trials with different random seeds), please run below command. Note that the "clothing1m/idv_prediction_vectors/x/noisy1m.npy" data used below is provided in above Google drive. Similarly, more trials can be ran by using different combinations of "noisy1m.npy" files and random seeds. 

Trial 1
```
python train_clothing1m.py --folder_log iter0 --seed 0 --idv_prediction_vector clothing1m/idv_prediction_vectors/1/noisy1m.npy clothing1m/idv_prediction_vectors/2/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy
```
Trial 2
```
python train_clothing1m.py --folder_log iter0 --seed 1 --idv_prediction_vector clothing1m/idv_prediction_vectors/19/noisy1m.npy clothing1m/idv_prediction_vectors/2/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy
```
Trial 3
```
python train_clothing1m.py --folder_log iter0 --seed 2 --idv_prediction_vector clothing1m/idv_prediction_vectors/1/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/19/noisy1m.npy
```

#### To run the code for Clothing1M in the second iteration (for 1 trial with random seed 0), run below command. Note that the weights "iter0/x/train_best.pth.tar" are abtained from the above runs with respective random seed x. Similarly, more trials can be ran by using different combinations of "train_best.pth.tar" files and random seeds. And the command to run the third iteration can be modified from below accordingly.
Trial 1
```
python train_clothing1m.py --folder_log iter1 --seed 0 --idv_weights iter0/0/train_best.pth.tar iter0/1/train_best.pth.tar iter0/2/train_best.pth.tar
```



    
NC instance identification task: The 200 manually verified NC instances from the Clothing1M dataset can be found in NC_instances.txt.
