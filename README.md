# edge-optimized-ship-detection
Edge-Optimized Ship Detection uses YOLO and RCNN models to detect ships in port areas from aerial images. Optimized for edge devices, it ensures real-time performance with reduced latency. This repository includes data preprocessing, model training, and deployment scripts.



# SHIP DETECTION

This project focuses on detecting ships in aerial images using deep learning models. We utilized Mask R-CNN for high accuracy and real-time performance. Below is a detailed explanation of the key components and processes involved in this project.

## Contents

### Data Preprocessing
- **Loading Data**: We began by loading the dataset containing aerial images of ships.
- **Data Augmentation**: Various augmentation techniques such as rotation, flipping, and scaling were applied to enhance the dataset and improve the model's generalization capabilities.
- **Annotation Processing**: The annotations were processed to match the input requirements of the Mask R-CNN model, including converting bounding box coordinates.

### Model Training
- **Mask R-CNN Model**: Mask R-CNN is an extension of Faster R-CNN that adds a branch for predicting segmentation masks on each Region of Interest (RoI). We configured and trained the Mask R-CNN model on our ship detection dataset, optimizing the parameters to achieve high detection accuracy.
- **Training Pipeline**: The training pipeline involved setting up the data generators, defining the model architecture, compiling the model with appropriate loss functions, and training it on the augmented dataset.
- **Evaluation**: Post-training, the model's performance was evaluated using metrics such as precision, recall, and F1-score. Visualizations of the detection results were also generated to qualitatively assess the model's performance.

### Inference
- **Loading Trained Model**: The trained Mask R-CNN model was loaded for inference.
- **Detecting Ships**: Inference was performed on new aerial images to detect ships. The model outputs bounding boxes and segmentation masks for each detected ship.
- **Visualization**: The detection results, including bounding boxes and masks, were visualized on the input images to provide a clear understanding of the model's performance.

## Technologies Used
- **Python**: The primary programming language for implementation.
- **TensorFlow/Keras**: Libraries used for building and training the Mask R-CNN model.
- **OpenCV**: Used for image processing and data augmentation.
- **Matplotlib**: For visualizing results and performance metrics.

## Project Structure
- `data/`: Directory for storing raw and processed data.
- `models/`: Directory for storing trained models.
- `src/`: Source code for data preprocessing, model training, and inference.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and prototyping.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

## Results
The model successfully detects ships in various conditions, showcasing robustness and accuracy.

# SHIP PORT DETECTION

This project implements an advanced ship port detection system using deep learning techniques. The goal is to accurately detect and identify ships in port areas from aerial images, optimized for deployment on edge devices to ensure real-time performance with reduced latency. This system utilizes Mask R-CNN models for high precision and efficiency.

## Contents

### Data Preprocessing
- **Loading Data**: The dataset containing aerial images of port areas with ships was loaded.
- **Data Augmentation**: To enhance the dataset and improve the model's generalization capabilities, augmentation techniques such as rotation, flipping, and scaling were applied.
- **Annotation Processing**: The annotations were processed to match the input requirements of the Mask R-CNN model, converting bounding box coordinates appropriately.

### Model Training
- **Mask R-CNN Model**: The Mask R-CNN model was configured and trained on the ship port detection dataset. This involved fine-tuning the model parameters to optimize detection accuracy for ships in port areas.
- **Training Pipeline**: The training pipeline involved setting up data generators, defining the model architecture, compiling the model with appropriate loss functions, and training it on the augmented dataset.
- **Evaluation**: The model's performance was evaluated using metrics such as precision, recall, and F1-score. Visualizations of the detection results were also generated for qualitative assessment.

### Edge Deployment
- **Optimization**: The trained Mask R-CNN model was optimized for deployment on edge devices, focusing on reducing latency and ensuring real-time performance.
- **Deployment Scripts**: Scripts were provided to deploy the model on edge devices, allowing for real-time ship port detection in practical applications.
- **Performance Testing**: The performance of the deployed model was tested in real-world scenarios to ensure it meets the requirements for real-time detection.

## Technologies Used
- **Python**: The core programming language for this project.
- **TensorFlow/Keras**: Deep learning libraries for building and training the Mask R-CNN model.
- **OpenCV**: For image processing and augmentation.
- **Docker**: Containerization for easy deployment and scalability.

## Project Structure
- `data/`: Directory for storing raw and processed data.
- `models/`: Directory for storing trained Mask R-CNN models.
- `src/`: Source code for data preprocessing, model training, and edge deployment.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and prototyping.
- `requirements.txt`: List of required Python packages.
- `Dockerfile`: Instructions for building a Docker image.
- `README.md`: Project documentation.

## Results
The model effectively detects ships in port areas, optimized for real-time performance on edge devices.


# EDGE DEPLOYMENT RESULTS
NVIDIA developed the Jetson Nano computer module, for edge AI applications offering a cost-compact solution. With its GPU architecture to that of other devices, it excels in executing deep learning algorithms. Industries like robotics, drones, IoT devices, and embedded systems commonly utilize it for AI processing without relying on cloud services. It serves as a platform for developers and enthusiasts to experiment develop and deploy AI applications in a compact form factor. 
      Getting YOLO set up for object detection on the Jetson Nano involves some steps. First, install required libraries like PyTorch and download trained YOLO weights. Create an environment to avoid any conflicts before running Python to load the model and prepare the input image for analysis. The real magic happens during inference when YOLO identifies objects. Finally convert the model output into a format such as bounding boxes overlaid on an image. For optimized performance, on the Jetson Nanos resources consider using a version of the advanced YOLO model or implementing quantization techniques for quicker processing speed.
     To create a more straightforward model that could be installed on the Jetson Nano and tuned for object detection, the proposed model used optimization techniques with TensorRT. The YOLO model is first changed to a lighter version that is more suited for the Jetson Nano's limited resources using PyTorch's torch.onnx.export() function. As a result, TensorRT is compatible with the model. After TensorRT is installed on the development system, it optimizes the ONNX model for Jetson Nano GPU inference. TensorRT performs improvements such as layer fusion, accurate calibration, and kernel auto-tuning to improve efficiency. Additionally, several quantization techniques are applied to reduce the model's size and expedite inference.  Both dynamic range and integer quantization are supported by TensorRT, which lowers the model's weights while guaranteeing faster processing with little degradation of accuracy. Lastly, the Jetson Nano is used to assess the performance of the optimized model. 

Table 4: Inference in different scenarios
Power Consumption	Inference Time
5W	36298.4ms
MaxN	16434.2ms

After analyzing the data from Table 4, it concluded that the system performs 2.2 times faster when operating in "MaxN" mode compared to the 5W mode. This difference is likely due to the system being able to draw power in "MaxN" mode leading to performance. On the hand carrying out inference tasks takes longer in low power settings, like the 5W mode.


![Screenshot from 2024-04-23 12-50-21](https://github.com/Pravallika030407/edge-optimized-ship-detection/assets/111449918/d47ee14f-94dc-417b-896d-2ed2922052ad)
![Screenshot from 2024-04-23 12-52-29](https://github.com/Pravallika030407/edge-optimized-ship-detection/assets/111449918/e1fedadb-5fa8-4575-ab7e-2b7ef7363c41)
![Screenshot from 2024-04-23 12-53-14](https://github.com/Pravallika030407/edge-optimized-ship-detection/assets/111449918/81fb3d8f-0641-4e0b-b65e-f1e1b11d4c2e)


