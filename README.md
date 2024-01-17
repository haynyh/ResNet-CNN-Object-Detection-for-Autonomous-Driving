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

![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/23b96b85-b943-40a5-9ad7-c70b9758174b)


![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/cc844302-3f9b-4174-a1d7-eb892be39317)



![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/8795b70a-0540-4ba9-a1d1-f18ea609d55a)


* Grid search (learning rate) <br />
![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/742ac65b-6e76-4c45-95b3-a7f81ff9a7f8)



![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/c2f88075-6591-4b9f-b93a-d68022073130)


* Grid search (batch size) <br />
![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/8030c387-ea48-4907-9fea-026f97195276)



![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/a10f2fd8-51d6-46d2-9ce0-5df8a4d17da6)


* Grid search (weight decay) <br />
![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/17c0a61d-f8cb-4a84-a691-df3ec3227ffc)



![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/44afc0f7-d08a-4cca-a90e-44297ca8622e)


* Survey Analysis of grid search <br />
![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/722fbd2c-7bb5-4c83-9f88-1f2ec9bb2e40)


<br />
### Prediction examples

![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/d38ccba1-1891-4d46-900a-ddd0e50c7200)

![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/b31d026c-bfcc-4ad5-a848-cac5966337df)


