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
results = model.train(data=data_config,
                        epochs=epochs,
                        imgsz=img_size,
                        batch=8,
                        lr0=0.01,          # initial learning rate
                        lrf=0.001,          # final OneCycleLR learning rate
                        momentum=0.937,    # SGD momentum / Adam beta1
                        weight_decay=0.0005,   # optimizer weight decay
                        warmup_epochs=3.0,     # number of warmup epochs
                        warmup_momentum=0.8,   # initial momentum during warmup
                        warmup_bias_lr=0.1,    # initial bias lr during warmup
                        box=7.5,           # box loss gain
                        cls=0.5,           # classification loss gain
                        dfl=1.5,           # distribution focal loss gain
                        iou=0.20,        # IoU training threshold
                        hsv_h=0.015,       # HSV hue augmentation
                        hsv_s=0.7,         # HSV saturation augmentation
                        hsv_v=0.4,         # HSV value augmentation
                        degrees=0.0,       # rotation (+/- deg)
                        translate=0.1,     # translation (+/- fraction)
                        scale=0.9,         # scale (+/- gain)
                        shear=0.0,         # shear (+/- deg)
                        perspective=0.0,   # perspective (+/- fraction)
                        flipud=0.0,        # up-down flip probability
                        fliplr=0.5,        # left-right flip probability
                        mosaic=1.0,        # mosaic augmentation probability
                        mixup=0.15,        # mixup augmentation probability
                        copy_paste=0.3,     # segment copy-paste augmentation probability
                        optimizer='AdamW',   # optimizer
)

# Optionally, you can log additional metrics or artifacts using ClearML APIs
# For example, logging the training results summary:
task.get_logger().report_text("Training completed successfully!")
print("Training completed!")
