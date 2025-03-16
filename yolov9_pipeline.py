from clearml import Task, Dataset
from clearml.automation.controller import PipelineDecorator
from clearml.automation import HyperParameterOptimizer, UniformParameterRange
from clearml.automation.optuna import OptimizerOptuna
from ultralytics import YOLO

# ------------------------
# STEP 1: Dataset versioning
# ------------------------
@PipelineDecorator.component(return_values=["dataset_id"])
def version_dataset():
    dataset = Dataset.create(
        dataset_name="YOLOv9_Dataset",
        dataset_project="YOLOv9_Training",
        dataset_tags=["version1"]
    )

    print("Adding dataset files...")
    dataset.add_files(path="./datasets/color_products_dataset")

    print("Uploading dataset files to ClearML storage...")
    dataset.upload()  # Ensure files are uploaded before finalizing

    print("Finalizing dataset version...")
    dataset.finalize()
    
    dataset_id = dataset.id

    print(f"Dataset version created: {dataset.id}")
    return dataset_id

# ------------------------
# STEP 2: Base training
# ------------------------
@PipelineDecorator.component(return_values=["base_task_id"])
def base_train_yolov9(dataset_id):
    # init ClearML task
    task = Task.init(
        project_name="YOLOv9_Training",
        task_name="Base_Train_YOLOv9",
        reuse_last_task_id=False
    )
    # link dataset version (for traceability)
    task.connect_configuration({"dataset_id": dataset_id})

    # define hyperparams
    params = {
        "data_config": "./data.yaml",
        "model_config": "./yolov9_architecture.yaml",
        "epochs": 1,
        "img_size": 640,
        "batch_size": 8,
    }
    task.connect(params)

    print("TRAINING PARAMS", params) 
    # Explicitly convert any numeric hyperparams to int:
    params["epochs"] = int(float(params["epochs"]))
    params["img_size"] = int(float(params["img_size"]))
    params["batch_size"] = int(float(params["batch_size"]))

    # train with ultralytics
    model = YOLO(params["model_config"])
    model.train(
        data=params["data_config"],
        epochs=params["epochs"],
        imgsz=params["img_size"],
        batch=params["batch_size"]
    )

    return task.id

# ------------------------
# STEP 3: Hyperparameter Tuning
# ------------------------
@PipelineDecorator.component()
def hyperparam_optimize(base_task_id):
    opt_task = Task.init(
        project_name="YOLOv9_Training",
        task_name="YOLOv9_HPO",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            UniformParameterRange("General/epochs", 2, 6, step_size=2),
            UniformParameterRange("General/img_size", 320, 640, step_size=160),
            UniformParameterRange("General/batch_size", 2, 8, step_size=2),
        ],
        # this is the objective metric we want to maximize/minimize
        objective_metric_title="metrics",
        objective_metric_series="mAP_0.5",
        # now we decide if we want to maximize it or minimize it (accuracy we maximize)
        objective_metric_sign="max",
        # let us limit the number of concurrent experiments,
        # this in turn will make sure we don't bombard the scheduler with experiments.
        # if we have an auto-scaler connected, this, by proxy, will limit the number of machine
        max_number_of_concurrent_tasks=1,
        # this is the optimizer class (actually doing the optimization)
        # Currently, we can choose from GridSearch, RandomSearch or OptimizerBOHB (Bayesian optimization Hyper-Band)
        optimizer_class=OptimizerOptuna,
        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=5,  # 5,
        compute_time_limit=None,
        total_max_jobs=20,
        min_iteration_per_job=None,
        max_iteration_per_job=None,
    )

    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()

    top_exps = optimizer.get_top_experiments(top_k=1)
    if not top_exps:
        print("No experiments found!")
        return

    best_exp = top_exps[0]
    best_exp_id = best_exp.id
    best_map = best_exp.get_last_scalar_metrics().get("metrics", {}).get("mAP_0.5", {}).get("last")
    print(f"Best experiment ID: {best_exp_id}, mAP={best_map}")

    # optionally download best weights
    best_exp_task = Task.get_task(task_id=best_exp_id)
    for artifact_name, artifact_obj in best_exp_task.artifacts.items():
        if "best" in artifact_name.lower() or "weights" in artifact_name.lower():
            local_path = artifact_obj.get_local_copy()
            print(f"Downloaded best model artifact '{artifact_name}' to {local_path}")

# ------------------------
# Pipeline flow function
# ------------------------
@PipelineDecorator.pipeline(
    name="YOLOv9_EndToEnd_Pipeline",
    project="YOLOv9_Training"
)
def run_pipeline():
    dataset_id = version_dataset()
    base_id = base_train_yolov9(dataset_id=dataset_id)
    hyperparam_optimize(base_task_id=base_id)

if __name__ == "__main__":
    print("Running YOLOv9 pipeline locally...")
    PipelineDecorator.run_locally()
    run_pipeline()
