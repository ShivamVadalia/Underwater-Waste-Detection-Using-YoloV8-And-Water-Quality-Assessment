
# Underwater Waste Detection Using YoloV8 And Water Quality Assessment

The project addresses the issue of growing underwater waste in oceans and seas. It offers three solutions: YoloV8 Algorithm-based underwater waste detection, a rule-based classifier for aquatic life habitat assessment, and a Machine Learning model for water classification as fit for use or not fit. The first model was trained on a dataset of 5000 images, while the second model used chemical properties guidelines from US EPA and WHO. The third model was trained on a dataset with over 6 million rows, providing reliable water classification results.




## Authors

- [@ShivamVadalia](https://github.com/ShivamVadalia)
- [@SiddharthChavan](https://github.com/SiddharthChavan23)
- [@KrunalBhere](https://github.com/MORGUE28)
- [@FaizanNabi](https://github.com/FaizanNabi-hub)
- [@ZuberPatel](https://github.com/zp14037)

## Demo
Underwater Waste Detection Using YoloV8

[Click here](https://universe.roboflow.com/neural-ocean/neural_ocean/model/3)


## Features

- Can detect underwater waste based on input images.
- Classifies water as potable or not based on chemical properties of water
- Classifies water as habitual for aquatic life or not.

## Architecture of YoloV8

<img width="600" alt="Architecture of YoloV8" src="https://blog.roboflow.com/content/images/size/w1000/2023/01/image-16.png">

## Tech Stack

- Python
- Dark Channel Prior Algorithm
- YoloV8(from Ultralytics)
- Xgboost-Classifier

## Screenshots

**Input Images:**

![test1](https://github.com/ShivamVadalia/Underwater-Waste-Detection-Using-YoloV8-And-Water-Quality-Assessment/assets/72978511/db31853b-61ad-4ebe-9215-64be12ca7c75)
![test2](https://github.com/ShivamVadalia/Underwater-Waste-Detection-Using-YoloV8-And-Water-Quality-Assessment/assets/72978511/641d0aff-2866-400d-8ba5-9c923a1ab944)

**Denoised Images After Running Dark Channel Prior Algorithm:**

![test1_dcp](https://github.com/ShivamVadalia/Underwater-Waste-Detection-Using-YoloV8-And-Water-Quality-Assessment/assets/72978511/63fb7d5b-9e00-4146-a284-cc5103fc02c0)
![test2_dcp](https://github.com/ShivamVadalia/Underwater-Waste-Detection-Using-YoloV8-And-Water-Quality-Assessment/assets/72978511/a2343278-fb5f-4a4f-afa7-1e87ad46385e)

**Output Images:**

![test1_out](https://github.com/ShivamVadalia/Underwater-Waste-Detection-Using-YoloV8-And-Water-Quality-Assessment/assets/72978511/a16af600-7fd7-4058-8270-bd0ce30b0f9c)
![test2_out](https://github.com/ShivamVadalia/Underwater-Waste-Detection-Using-YoloV8-And-Water-Quality-Assessment/assets/72978511/bf5196a2-44bf-4bf9-980a-1b046e81b798)

## Datasets Used
- [Underwater Trash Images](https://universe.roboflow.com/neural-ocean/neural_ocean)
- [Water Quality](https://www.kaggle.com/datasets/naiborhujosua/predict-the-quality-of-freshwater)

## Acknowledgements

 - [YoloV8 Training Notebook by Roboflow](https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb)
 - [Architecture of YoloV8](https://blog.roboflow.com/whats-new-in-yolov8/)
 - [Single Image Haze Removal Using Dark Channel Prior](https://projectsweb.cs.washington.edu/research/insects/CVPR2009/award/hazeremv_drkchnl.pdf)
