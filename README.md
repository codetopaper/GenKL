# GenKL: An iterative framework for resolving label ambiguity and label non-conformity in web images via a new Generalized KL divergence

## The GenKL code for Clothing1M, Food101/Food101N, and mini WebVision 1.0 datasets are available here.

#### To run the code for Clothing1M in the first iteration (for 3 trials with different random seeds), please run below command. Note that the "clothing1m/idv_prediction_vectors/x/noisy1m.npy" data used below is provided [here](https://drive.google.com/drive/folders/1dP4m61BTNWMN-9vVJIqWZvmsoTWD3Syd?usp=sharing). Similarly, more trials can be ran by using different combinations of "noisy1m.npy" files and random seeds. 

In iteration 0, the NC instances are identified via the averaged softmax vectors from models trained on the clean 50k of clothing1m. To train one model on the clean 50k of clothing1m to produce each individual softmax vectors, run below code:
```
python train_clothing1m.py --folder_log iter0 --seed 0 --idv_prediction_vector clothing1m/idv_prediction_vectors/1/noisy1m.npy clothing1m/idv_prediction_vectors/2/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy
```

Trial 1
```
python clothing1m_iter0_softmax_weights.py --seed 0 
```
Trial 2
```
python train_clothing1m.py --folder_log iter0 --seed 1 --idv_prediction_vector clothing1m/idv_prediction_vectors/19/noisy1m.npy clothing1m/idv_prediction_vectors/2/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy
```
Trial 3
```
python train_clothing1m.py --folder_log iter0 --seed 2 --idv_prediction_vector clothing1m/idv_prediction_vectors/1/noisy1m.npy clothing1m/idv_prediction_vectors/9/noisy1m.npy clothing1m/idv_prediction_vectors/4/noisy1m.npy clothing1m/idv_prediction_vectors/7/noisy1m.npy clothing1m/idv_prediction_vectors/19/noisy1m.npy
```

#### To run the code for Clothing1M in the second iteration (for 1 trial with random seed 0), run below command. Note that the weights "iter0/x/train_best.pth.tar" are abtained from the above runs in the first iteration with respective random seed x. Similarly, more trials can be ran by using different combinations of "train_best.pth.tar" files and random seeds. And the command to run the third iteration can be modified from below accordingly.
Trial 1
```
python train_clothing1m.py --folder_log iter1 --seed 0 --idv_weights iter0/0/train_best.pth.tar iter0/1/train_best.pth.tar iter0/2/train_best.pth.tar
```


#### To run the code for Food101/Food101N, please run below command. Note that the "food/idv_weights/x/ckpt_b_val.pth" data used below is provided [here](https://drive.google.com/drive/folders/1dP4m61BTNWMN-9vVJIqWZvmsoTWD3Syd?usp=sharing). Similarly, more trials can be ran by using different combinations of "ckpt_b_val.pth" files and random seeds. 


```
python train_Food101N.py --seed 0 --idv_weights food/idv_weights/1/ckpt_b_val.pth food/idv_weights/2/ckpt_b_val.pth food/idv_weights/3/ckpt_b_val.pth food/idv_weights/4/ckpt_b_val.pth food/idv_weights/4/ckpt_b_val.pth
```

#### To run the code for mini WebVision 1.0, please run below command. Note that the "mini_WebVision_1.0/10ep/idv_weights/x/checkpoint_0009.pth.tar" and "mini_WebVision_1.0/avg_prediction_vectors/x/avg.npy" data used below is provided [here](https://drive.google.com/drive/folders/1dP4m61BTNWMN-9vVJIqWZvmsoTWD3Syd?usp=sharing). Similarly, more trials can be ran by using different combinations of "checkpoint_0009.pth.tar" and "avg.npy" files and random seeds. 
```
python train_miniwebvision.py --index 0 --weights mini_WebVision_1.0/10ep/idv_weights/0/checkpoint_0009.pth.tar --avg_prediction_vector mini_WebVision_1.0/avg_prediction_vectors/1/avg.npy
```


## NC instance identification task: The 200 manually verified NC instances from the Clothing1M dataset can be found in NC_instances.txt.
