# **RoadSight: Real-Time Vehicle Detection with YOLOv8**

## **Overview**

**RoadSight** is a real-time vehicle detection system designed to enhance road safety and traffic management. Leveraging the power of advanced deep learning techniques, this project utilizes the YOLOv8 (You Only Look Once) object detection model to accurately identify and locate vehicles in various driving conditions. RoadSight aims to provide efficient and reliable vehicle detection for applications such as autonomous driving, traffic monitoring, and intelligent transportation systems.

## **Technology Stack**

- **Python**: The primary programming language used for implementing the detection system.
- **YOLOv8**: A cutting-edge, real-time object detection algorithm that offers high accuracy and speed.
- **OpenCV**: An open-source computer vision library used for image processing and manipulation.
- **PyTorch**: A deep learning framework employed for training and deploying the YOLOv8 model.
- **NumPy**: A library for efficient numerical computations and array manipulations.
- **Matplotlib**: Used for visualizing detection results and performance metrics.

## **Project Structure**

### **Data Preparation**

- Collected a diverse dataset comprising images of roads with various types and numbers of vehicles under different weather and lighting conditions.
- Annotated the dataset to include bounding boxes around vehicles, ensuring the model learns precise localization.
- Performed data augmentation techniques such as scaling, rotation, and flipping to enhance the model's robustness and generalization capabilities.
- Split the dataset into training, validation, and testing sets to evaluate the model's performance effectively.

### **Model Training**

- Configured the YOLOv8 model with appropriate hyperparameters tailored to the vehicle detection task.
- Trained the model using the prepared dataset, employing techniques like learning rate scheduling and early stopping to optimize performance.
- Monitored training progress through loss metrics and validation accuracy to prevent overfitting and ensure model convergence.
- Conducted iterative evaluations and fine-tuning to achieve the best balance between detection accuracy and inference speed.

### **Evaluation and Testing**

## Train/Test Result:
![results](https://github.com/user-attachments/assets/010a93bb-5a61-4b5f-bf97-f3cb023d87df)


## Confusion Matrix:
![confusion_matrix](https://github.com/user-attachments/assets/42d3fe79-02f5-4169-b613-1b78365980b5)

- Tested the trained model on unseen data to assess its real-world performance in detecting and localizing vehicles accurately.
- Evaluated metrics such as Precision, Recall, and mAP (mean Average Precision) to quantify detection performance.
- Analyzed the model's performance across different scenarios, including varying traffic densities and environmental conditions.
- Visualized detection results by overlaying bounding boxes on test images and videos to qualitatively assess accuracy.

## **Results**
![val_batch1_pred](https://github.com/user-attachments/assets/309d637a-0856-459f-92cd-dbaad3001004)

RoadSight successfully demonstrates high accuracy and efficiency in real-time vehicle detection tasks:

- **Accuracy**: Achieved a high mAP score, indicating reliable detection and localization of vehicles across diverse scenarios.
- **Speed**: Demonstrated real-time processing capabilities, making it suitable for applications requiring instantaneous detection.
- **Robustness**: Maintained consistent performance under varying conditions such as different lighting, weather, and traffic densities.
- **Scalability**: The model's architecture allows for easy scaling and adaptation to detect other object classes with additional training.

## **Conclusion**

RoadSight showcases the effective integration of advanced deep learning models for practical and critical applications in road safety and traffic management. The project's success underscores the potential of leveraging state-of-the-art object detection algorithms like YOLOv8 to develop systems that can significantly contribute to reducing road accidents and improving traffic flow. Future enhancements may include extending the model to detect other road entities such as pedestrians, cyclists, and traffic signs, further broadening its applicability in intelligent transportation systems.
