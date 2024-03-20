# SegCropNet
SegCropNet is a state-of-the-art agricultural navigation model that greatly improves the robustness of the crop row detection task in specific scenarios as well as the accuracy of crop row line detection.

## Network Architecture
We propose a dual-branch network to better train machines to recognize ROI regions.One branch is used for semantic segmentation and the other for instance segmentation.Instance segmentation and binary segmentation are two independent branches that have no influence on each other. It's just that at the end of the cluster, we only cluster the crop lines in binary segmentation.

![path](https://p.ipic.vip/0u7o5y.png)
## Crop image processing results
In crop line detection and fitting, we comprehensively consider perspective distortion, road curvature and image quality, and obtain more robust and accurate crop line detection results
![image path2](https://p.ipic.vip/j7ciln.jpeg)

## Results of different models(Average distance deviation from ground truth (pixels))
![image path3](https://p.ipic.vip/3qaevj.jpeg)
## Experiments
You can now test our SegCropNet by:
```
python test.py --img ./imgs/pic_1.jpg --model_type ENet --model ./log/best_model.pth --width 512 --height 256 --save ./test_output  
```

or you can train it by yourself as follows:
```
python train.py --dataset ./data/sets --model_type ENet --lose_type FocalLoss --save ./log --json config.json
```
