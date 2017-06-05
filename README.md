**Deformable ConvNets** is initially described in an [arxiv tech report](https://arxiv.org/abs/1703.06211).

**R-FCN** is initially described in a [NIPS 2016 paper](https://arxiv.org/abs/1605.06409).

**Soft-NMS** is initially described in an [arxiv tech report](https://arxiv.org/abs/1704.04503).

Our goal was to test Soft-NMS with a state-of-the-art detector, so Deformable-R-FCN was trained on 800x1200 size images with 15 anchors. Multi-Scale testing was also added with 6 scales. Union of all boxes at each scale was computed before performing NMS. Please note that the repository does not include the scripts for multi-scale testing as I just cache the boxes for each different scale and do NMS separately. The scales used in multi-scale testing were as follows, [(480, 800), (576,900), (688, 1100), (800,1200), (1200, 1600), (1400, 2000)]. 

The trained model can be downloaded from [here](https://drive.google.com/file/d/0B6T5quL13CdHZ3ZrRVNjcnFmZk0).

|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> | <sub>Recall</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|-------|
| <sub>Baseline D-R-FCN</sub> | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 35.7 | 56.8    | 38.3    | 15.2  | 38.8  | 51.5  |
| <sub>D-R-FCN, ResNet-v1-101, NMS</sub> | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 37.4 | 59.6    | 40.2    | 17.8  | 40.6  | 51.4  | 48.3  |
| <sub>D-R-FCN, ResNet-v1-101, SNMS</sub> | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 38.4 | 60.1    | 41.6    | 18.5  | 41.6  | 52.5  | 53.8  |
| <sub>D-R-FCN, ResNet-v1-101, MST, NMS</sub> | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 39.8 | 62.4    | 43.3    | 22.6  | 42.3  | 52.2  | 52.9  |
| <sub>D-R-FCN, ResNet-v1-101, MST, SNMS</sub> | <sub>coco trainval</sub> | <sub>coco test-dev</sub> | 40.9 | 62.8    | 45.0    | 23.3  | 43.6  | 53.3  | 60.4  |


