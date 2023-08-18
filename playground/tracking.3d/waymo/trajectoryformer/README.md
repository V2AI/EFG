
# TrajectoryFormer: 3D Object Tracking Transformer with Predictive Trajectory Hypotheses (ICCV 2023)

## Preprocessing
Please follow the command at [Data Preprocessing Documentations](../../../../README.md) to process Waymo Open Dataset and install `EFG`.

The preprocessed detection boxes of [CenterPoint](https://arxiv.org/abs/2006.11275) and [MPPNet](https://arxiv.org/abs/2205.05979) can be download form [Google Drive](https://drive.google.com/drive/folders/1SPbkf04DxB3brCKnM3N8_4XgWTOfNMyF?usp=sharing) and put the download files in the `EFG/datasets/waymo/` folder.

Finally, compile the evaluation metrics tool provided by Waymo officials by following [Quick Guide to Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset/blob/v1.5.0/docs/quick_start.md).

## Training Motionpredictor

```bash
cd playground/tracking.3d/waymo/trajectoryformer.motionpred;
efg_run --nug-gpus 8 task train
```
The trained model will be saved at `./log/model_final.pth`.

## Training TrajectoryFormer

```bash
# CenterPoint
cd playground/tracking.3d/waymo/trajectoryformer.centerpoint;
efg_run --num-gpus 8 task train 

# MPPNet 
cd playground/tracking.3d/waymo/trajectoryformer.mppnet;
efg_run --num-gpus 8 task train 
```

## Evaluation
Set the metrics tool path, such as `/home/user/bazel_bin/waymo_open_dataset/metrics/tools/compute_tracking_main` to the yaml. \
For CenterPoint,
```bash
cd playground/tracking.3d/waymo/trajectoryformer.centerpoint;
# for vehicle
efg_run --num-gpus 8 task val \
trainer.eval_metrics_path /path/to/your/tools 
model.nms_thresh 0.1
model.eval_class VEHICLE

# for pedestrian
efg_run --num-gpus 8 task val \
trainer.eval_metrics_path /path/to/your/tools 
model.nms_thresh 0.7 \
model.eval_class PEDESTRIAN

# for cyclist
efg_run --num-gpus 8 task val \
trainer.eval_metrics_path /path/to/your/tools 
model.nms_thresh 0.7 \
model.eval_class CYCLIST
```

For MPPNet,
```bash
# eval vehicle, pedestrian, cyclist
cd playground/tracking.3d/waymo/trajectoryformer.mppnet;
efg_run --num-gpus 8 task val \
trainer.eval_metrics_path /path/to/your/tools
model.eval_class VEHICLE or PEDESTRIAN or CYCLIST
```
If you want the pretrained model, please contact `chenxuesong@link.cuhk.edu.hk`.

## Citation
```
@article{chen2023trajectoryformer,
        title={TrajectoryFormer: 3D Object Tracking Transformer with Predictive Trajectory Hypotheses},
        author={Chen, Xuesong and Shi, Shaoshuai and Zhang, Chao and Zhu, Benjin and Wang, Qiang and Cheung, Ka Chun and See, Simon and Li, Hongsheng},
        journal={arXiv preprint arXiv:2306.05888},
        year={2023}

@inproceedings{chen2022mppnet,
  title={Mppnet: Multi-frame feature intertwining with proxy points for 3d temporal object detection},
  author={Chen, Xuesong and Shi, Shaoshuai and Zhu, Benjin and Cheung, Ka Chun and Xu, Hang and Li, Hongsheng},
  booktitle={European Conference on Computer Vision},
  pages={680--697},
  year={2022},
  organization={Springer}
}

}
```
