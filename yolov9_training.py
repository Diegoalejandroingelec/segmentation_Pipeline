# train_yolov9_clearml.py

from ultralytics import YOLO
from clearml import Task


# Initialize ClearML task (this will automatically log metrics, parameters, and artifacts)
task = Task.init(project_name='YOLOv9_Training', task_name='Train_YOLOv9_with_ClearML')

# Define training parameters
data_config = './data.yaml'  # YAML file with dataset configuration (train/val paths, classes, etc.)
model_config = './yolov9_architecture.yaml'                # Your YOLOv9 configuration file (ensure this file exists or modify accordingly)
epochs = 1                                 # Number of training epochs
img_size = 640                              # Image size (can adjust based on your requirements)

# Create the YOLO model (this uses the configuration file provided)
model = YOLO(model_config)

# Start training with the defined parameters.
# Ultralytics will automatically log training progress, and ClearML will capture these metrics.
results = model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=8)

# Optionally, you can log additional metrics or artifacts using ClearML APIs
# For example, logging the training results summary:
task.get_logger().report_text("Training completed successfully!")
print("Training completed!")
