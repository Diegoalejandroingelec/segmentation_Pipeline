from clearml import Task, Dataset
from clearml.automation.controller import PipelineDecorator
from clearml.automation import HyperParameterOptimizer, UniformParameterRange
from clearml.automation.optuna import OptimizerOptuna
from ultralytics import YOLO

# --------------------------------------------------
# STEP 1: Dataset Versioning
# --------------------------------------------------
@PipelineDecorator.component(return_values=["dataset_id"])
def version_dataset():
    """
    1) Creates (or updates) a dataset version in ClearML
    2) Returns the dataset ID, which other steps can use or log
    """
    dataset_project = "YOLOv9_Training"
    dataset_name = "YOLOv9_Dataset"
    dataset_path = "./dataset"  # local path to your dataset

    dataset = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        dataset_tags=["version1"]
    )
    dataset.add_files(path=dataset_path)
    # Optional: dataset.upload(output_url="s3://your_bucket/datasets")

    dataset.finalize()
    print(f"[Dataset Versioning] Dataset version created: {dataset.id}")
    return dataset.id

# --------------------------------------------------
# STEP 2: Base YOLOv9 Training
# --------------------------------------------------
@PipelineDecorator.component(return_values=["base_task_id"])
def base_train_yolov9(dataset_id):
    """
    1) Initializes a ClearML task for YOLOv9
    2) Trains a base YOLOv9 model
    3) Returns the newly created task ID so we can reference it later
    """
    # Initialize the training task
    task = Task.init(
        project_name="YOLOv9_Training",
        task_name="Base_Train_YOLOv9_Pipeline",
        reuse_last_task_id=False
    )

    # Connect dataset info (so we can trace which dataset version was used)
    # This isn't mandatory, but good for traceability:
    task.connect_configuration({"dataset_id": dataset_id})

    # Define training parameters
    params = {
        "data_config": "./data.yaml",               # Points to the dataset config
        "model_config": "./yolov9_architecture.yaml",
        "epochs": 50,
        "img_size": 640,
        "batch_size": 4,
    }
    task.connect(params)  # Logs hyperparameters to ClearML

    # Train YOLOv9
    model = YOLO(params["model_config"])
    _results = model.train(
        data=params["data_config"],
        epochs=params["epochs"],
        imgsz=params["img_size"],
        batch=params["batch_size"],
    )

    task.get_logger().report_text("Base YOLOv9 training completed!")
    print("[Base Train YOLOv9] Training completed!")

    return task.id

# --------------------------------------------------
# STEP 3: Hyperparameter Optimization
# --------------------------------------------------
@PipelineDecorator.component()
def hyperparam_optimize(base_task_id):
    """
    1) Uses the base_task_id from step 2 as a template for HPO
    2) Runs multiple experiments with different hyperparams
    3) Prints the best run and downloads its best model artifact
    """
    # Initialize an optimizer task
    opt_task = Task.init(
        project_name="YOLOv9_Training",
        task_name="YOLOv9_HPO_Pipeline",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            # Param paths must match your training script's structure in ClearML UI
            UniformParameterRange("epochs", min_value=20, max_value=60, step_size=20),
            UniformParameterRange("img_size", min_value=320, max_value=640, step_size=160),
            UniformParameterRange("batch_size", min_value=2, max_value=6, step_size=2),
        ],
        objective_metric_title="metrics",
        objective_metric_series="mAP_0.5",
        objective_metric_sign="max",   # YOLO's mAP -> maximize
        max_number_of_concurrent_tasks=1,
        optimizer_class=OptimizerOptuna,
        save_top_k_tasks_only=3,
        total_max_jobs=5,
    )

    optimizer.set_time_limit(in_minutes=120)  # 2 hour time limit
    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()

    print("[HPO] Hyperparameter optimization finished!")

    # Find the best experiment
    top_experiments = optimizer.get_top_experiments(top_k=1)
    if not top_experiments:
        print("[HPO] No experiments found.")
        return

    best_experiment = top_experiments[0]
    best_experiment_id = best_experiment.id
    best_map = best_experiment.get_last_scalar_metrics().get("metrics", {}).get("mAP_0.5", {}).get("last")
    print(f"[HPO] Best experiment ID: {best_experiment_id}, mAP_0.5={best_map}")

    # Download the best model artifact if YOLO saved it
    best_experiment_task = Task.get_task(task_id=best_experiment_id)
    artifacts = best_experiment_task.artifacts
    for artifact_name, artifact_obj in artifacts.items():
        if "best" in artifact_name.lower() or "weights" in artifact_name.lower():
            local_path = artifact_obj.get_local_copy()
            print(f"[HPO] Downloaded best model artifact '{artifact_name}' to: {local_path}")

# --------------------------------------------------
# ASSEMBLE THE PIPELINE
# --------------------------------------------------
@PipelineDecorator.pipeline(
    name="YOLOv9_EndToEnd_Pipeline",
    project="YOLOv9_Training",
    version="1.0"
)
def run_pipeline():
    """
    Orchestrates three steps:
      1) version_dataset
      2) base_train_yolov9
      3) hyperparam_optimize
    """
    dataset_id = version_dataset()
    base_task_id = base_train_yolov9(dataset_id=dataset_id)
    hyperparam_optimize(base_task_id=base_task_id)

if __name__ == "__main__":
    # Execute pipeline locally
    print("Running YOLOv9 pipeline locally...")
    PipelineDecorator.run_locally()
