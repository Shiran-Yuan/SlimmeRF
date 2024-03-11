# SlimmeRF: Slimmable Radiance Fields (3DV'24 Oral)
Shiran Yuan and Hao Zhao

*AIR, Tsinghua University*

![logo](https://github.com/Shiran-Yuan/SlimmeRF/assets/105504535/1829fd23-32c4-45b0-a815-f6213e978614)

arXiv Paper Link: [2312.10034](https://arxiv.org/abs/2312.10034)

## Installing Requirements
```
conda create -n SlimmeRF python
conda activate SlimmeRF
pip install -r requirements.txt
```

## Datasets
Please arrange the datasets in a folder named "data". The following are the download links to the datasets:
+ [NeRF Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (Link to Google Drive)
+ [LLFF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) (Link to Google Drive)
+ [Tanks & Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip) (Download will start after clicking on link)

We note that we use a processed version of Tanks & Temples, and the `intrinsics.txt` file of "Ignatius" was incompatible with our code. Please manually replace it with the following contents:
```
1166.564936839068 0.0 960.0 0.0
0.0 1166.564936839068 540.0 0.0
0.0 0.0 1.0 0.0
0.0 0.0 0.0 1.0
```

After installation, the structure of the `data` directory should be as follows:  
<details>
  <summary>
    Click to Unfold
  </summary>

  ```
  data/  
  ├─ nerf_llff_data/  
  │  ├─ fern/  
  │  ├─ flower/  
  │  ├─ fortress/  
  │  ├─ horns/  
  │  ├─ leaves/  
  │  ├─ orchids/  
  │  ├─ room/  
  │  ├─ trex/  
  ├─ nerf_synthetic/  
  │  ├─ chair/  
  │  ├─ drums/  
  │  ├─ ficus/  
  │  ├─ hotdog/  
  │  ├─ lego/  
  │  ├─ materials/  
  │  ├─ mic/  
  │  ├─ ship/  
  ├─ TanksAndTemple/  
  │  ├─ Barn/  
  │  ├─ Caterpillar/  
  │  ├─ Family/  
  │  ├─ Ignatius/  
  │  ├─ Truck/  
  ```
</details>

## Training
Models are trained with config files using the following command:
```
python train.py --config configs/nerf_synthetic/hotdog.txt
```
The variable `datadir` in the config file controls which scene is used. `expname` controls the folder in which the results are stored.

The hyper-parameters `upsilon` ($\upsilon$) and `eta` ($\eta$) can be controlled from the file `train.py`.

To control which after-slimming ranks to test, use the `test_slimmed` variable in `train.py`. 

## Results
Please create an empty folder `log` to store the results. Results of experiments are be stored in `log/[expname]`. 

The model (in its full form) is stored in `log/[expname].th`. For an actually applicable model, do the following steps:

1. Extract the `state_dict` from the `.th` file.
2. Remove all masks (`density_line_mask`, `app_line_mask`, `density_plane_mask`, `app_plane_mask`).
3. Convert the remaining parameter tensors to `torch.float16`. (Optional)

Then, when slimming is needed, directly truncate all tensors along their rank dimension.

2D synthesis testing results are directly given in the `log` folder. For rendering use the following:
```
python train.py --config configs/nerf_synthetic/hotdog.txt --ckpt path/to/checkpoint --render_only 1 --render_test 1
```
For mesh extraction use the following:
```
python train.py --config configs/nerf_synthetic/hotdog.txt --ckpt path/to/checkpoint --export_mesh 1
```

## Acknowledgements
Our code is partially based on the codebase of [TensoRF](https://github.com/apchenstu/TensoRF). We would like to thank the authors of that work.
