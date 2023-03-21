# fcos.res50.fpn.coco.800size.1x  

seed: 37242394

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.575
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.323
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.571
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.369
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.802 | 57.539 | 42.066 | 22.952 | 42.786 | 49.809 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 53.267 | bicycle      | 28.881 | car            | 41.935 |  
| motorcycle    | 40.296 | airplane     | 64.162 | bus            | 64.231 |  
| train         | 59.074 | truck        | 34.161 | boat           | 23.247 |  
| traffic light | 25.749 | fire hydrant | 65.123 | stop sign      | 63.712 |  
| parking meter | 41.736 | bench        | 19.430 | bird           | 33.906 |  
| cat           | 65.326 | dog          | 62.083 | horse          | 52.277 |  
| sheep         | 50.622 | cow          | 56.500 | elephant       | 63.260 |  
| bear          | 71.055 | zebra        | 68.160 | giraffe        | 64.253 |  
| backpack      | 15.101 | umbrella     | 36.936 | handbag        | 13.466 |  
| tie           | 29.457 | suitcase     | 35.544 | frisbee        | 64.349 |  
| skis          | 17.711 | snowboard    | 28.126 | sports ball    | 45.110 |  
| kite          | 39.252 | baseball bat | 25.923 | baseball glove | 34.427 |  
| skateboard    | 47.973 | surfboard    | 30.812 | tennis racket  | 44.958 |  
| bottle        | 35.026 | wine glass   | 33.684 | cup            | 40.921 |  
| fork          | 27.445 | knife        | 14.667 | spoon          | 13.364 |  
| bowl          | 39.377 | banana       | 23.971 | apple          | 20.577 |  
| sandwich      | 32.532 | orange       | 31.783 | broccoli       | 23.171 |  
| carrot        | 17.947 | hot dog      | 29.371 | pizza          | 49.780 |  
| donut         | 47.162 | cake         | 33.933 | chair          | 25.585 |  
| couch         | 39.526 | potted plant | 26.820 | bed            | 38.489 |  
| dining table  | 25.032 | toilet       | 56.121 | tv             | 52.317 |  
| laptop        | 54.500 | mouse        | 58.372 | remote         | 26.979 |  
| keyboard      | 45.012 | cell phone   | 33.665 | microwave      | 57.897 |  
| oven          | 30.745 | toaster      | 29.127 | sink           | 33.379 |  
| refrigerator  | 49.711 | book         | 12.678 | clock          | 48.451 |  
| vase          | 36.081 | scissors     | 21.261 | teddy bear     | 45.035 |  
| hair drier    | 7.431  | toothbrush   | 13.666 |                |        |
