<div align="center">
<h1> AutoAlignV2 </h1>
<h3>Deformable Feature Aggregation for Dynamic Multi-Modal 3D Object Detection</h3>
<br>Zehui Chen, Zhenyu Li, Shiquan Zhang, Liangji Fang, Qinhong Jiang, Feng Zhao. 
<br>

<div><a href="https://arxiv.org/abs/2207.10316">[ECCV 2022 Paper] </a></div> 

<center>
<img src='figs/framework.png'>
</center>

</div>

## Performance

### nuScenes Val set
| Model | config | mAP | NDS |
| - | - | - | - |
| Baseline 1/8 subset | | - | - |
| AutoAlignV2 1/8 subset | | 58.5 | 63.2 |

### nuScenes Test Leaderboard
| Model | mAP | NDS |
| -|-|-|
| AutoAlign |  65.8  |  70.9  |
| AutoAlignV2 | 68.4 | 72.4|

## Note

The code is released without rechecking. We will clean up and recheck the code recently.

## Get Started

### Install Deformable Ops from DeformDETR

```
cd ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

### Prepare Dataset
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0
```

### Train Model
```
./tools/dist_train.sh aav2_cfg/centerpoint_voxel_nus_8subset_bs4_img1_nuimg_detach_deform_multipts.py 8
```

### Test Model
```
./tools/dist_test.sh aav2_cfg/centerpoint_voxel_nus_8subset_bs4_img1_nuimg_detach_deform_multipts.py work_dirs/centerpoint_voxel_nus_8subset_bs4_img1_nuimg_detach_deform_multipts/epoch_20.pth 8 --eval bbox
```

### Pretrained Model Weights
| Model | Weights |
| - | -|
|Pretrained YOLOX-640 | https://drive.google.com/file/d/1DuA16ocF8Dmfl0nC3NNSKG_Yqq-aXuLH/view?usp=sharing |
|Pretrained YOLOX-800 | https://drive.google.com/file/d/1YPSjG38qXrV8gYlq5FuhQCjfRIUvhaju/view?usp=sharing |
| AutoAlignV2 1/8 subset| https://drive.google.com/file/d/1F_B4bq5Y6QdkUHQbqeLaEGl-IZHHe1WB/view?usp=sharing |


## Citation
If you find our work useful for your research, please consider citing the paper
```
@article{chen2022autoalignv2,
  title={AutoAlignV2: Deformable Feature Aggregation for Dynamic Multi-Modal 3D Object Detection},
  author={Chen, Zehui and Li, Zhenyu and Zhang, Shiquan and Fang, Liangji and Jiang, Qinhong and Zhao, Feng},
  journal={ECCV},
  year={2022}
}
```