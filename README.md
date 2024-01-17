ResNet CNN: Object Detection for Autonomous Driving <br />


## Repo structure

* [data/](data/): dataset initiation, data augmentation
* [model/](model/): ResNet model architecture, convolutional blocks and layers
* [utils/](utils/): yolov1 loss function, image non maximimum suppression and inference
* [vis_result/](vis_result/): images of the prediction output


## Main files

* [train.py](train.py): main python file for training the CNN
* [predict.py](predict.py): main python file for inferencing the model with trained param and saving to vis_result
* [eval.py](eval.py): main python file for running model evaluation, adjusting threshold and calculating mAP

## Quick start and reproducing training and prediction


* Run [train.py --num_epochs _ --batch_size _ --learning_rate _ --output_dir _] , the trained param will be stored in --output_dir
* Run [eval.py --model_path _ --dataset_root _ --output_file _] , model_path to load your trained param, dataset_root to store images for evaluation on and output_file store the evaluation result
* Run [predict.py --model_path _ --image_path _ --vis_dir _] , model_path to load your trained param, image_path to store the images you want to predict on and vis_dir store the predicted images



### Model tuning process

Epoch lr       bs
1-10  1.00E-04 10
11-16 8.00E-05 7
16-20 6.00E-05 5
21-24 4.00E-05 3
25-40 2.00E-05 2
40-50 1.00E-05 2

Epoch lr       bs tloss Vloss mAP
10    1.00E-04 10 3.729 4.038 0.29
11-16 8.00E-05 7  3.376 3.946
16-20 6.00E-05 5  3.142 3.668
21-24 4.00E-05 3  2.917 3.732
25-40 2.00E-05 2  2.607 3.548
40-50 1.00E-05 2  2.498 3.539 0.36


![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/8795b70a-0540-4ba9-a1d1-f18ea609d55a)


* Grid search (learning rate)
Epoch lr       bs tloss Vloss    mAP      Weight decay
30    1.00E-04 10 2.941 3.518927 0.366528 5.00E-06
30    8.00E-05 10 2.818 3.428842 0.381609 5.00E-06
30    5.00E-05 10 2.834 3.499203 0.379528 5.00E-0


![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/c2f88075-6591-4b9f-b93a-d68022073130)


* Grid search (batch size)
Epoch lr       bs tloss Vloss    mAP      Weight decay
30    8.00E-05 10 2.818 3.428842 0.381609 5.00E-06
30    8.00E-05 5  2.895 3.447548 0.385097 5.00E-06


![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/a10f2fd8-51d6-46d2-9ce0-5df8a4d17da6)


* Grid search (weight decay)
Epoch lr       bs tloss Vloss    mAP      Weight decay
30    8.00E-05 5  2.895 3.447548 0.385097 5.00E-06
30    8.00E-05 5  2.654 3.298391 0.404869 5.00E-01


![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/44afc0f7-d08a-4cca-a90e-44297ca8622e)


* Survey Analysis of grid search
Epoch lr       bs tloss Vloss    mAP      Weight decay
30    1.00E-04 10 2.941 3.518927 0.366528 5.00E-06
30    8.00E-05 10 2.818 3.428842 0.381609 5.00E-06
30    5.00E-05 10 2.834 3.499203 0.379528 5.00E-06
30    8.00E-05 5  2.895 3.447548 0.385097 5.00E-06
30    8.00E-05 5  2.654 3.298391 0.404869 5.00E-01
60    8.00E-05 5  2.57  3.263978 0.417826 5.00E-01


### Prediction examples

![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/d38ccba1-1891-4d46-900a-ddd0e50c7200)

![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/b31d026c-bfcc-4ad5-a848-cac5966337df)


