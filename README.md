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



### Model training process

*Training without regularization
![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/11b230b9-7364-4549-95d9-8f1be71513c2)


![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/73128d7c-18b4-4cd3-a2aa-54f253bfab38)


*Training with regularization
![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/981a5c05-b6e1-4a89-bca3-56029b328a05)


![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/b6d2ce17-c38f-463f-9d7f-d535bc4afb7f)




<br />
### Prediction examples

![image](https://github.com/haynyh/ResNet-CNN-Object-Detection-for-Autonomous-Driving/assets/46237598/772d1188-df1d-4714-a5f1-b712d655a632)



