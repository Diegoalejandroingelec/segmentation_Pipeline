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
    from clearml import Task

    # 1) Initialize the ClearML Task
    task = Task.init(
        project_name="YOLOv9_Training",
        task_name="Base_Train_YOLOv9",
        reuse_last_task_id=False
    )
    task.connect_configuration({"dataset_id": dataset_id})

    # 2) Define default hyperparams
    default_params = {
        "data_config": "./data.yaml",
        "model_config": "./yolov9_architecture.yaml",
        "epochs": 20,
        "img_size": 640,
        "batch_size": 8,
        "lr0": 0.01,          # initial learning rate
        "lrf": 0.001,         # final OneCycleLR learning rate
        "momentum": 0.937,    # SGD momentum / Adam beta1
        "weight_decay": 0.001,   # optimizer weight decay
        "warmup_epochs": 3.0,     # number of warmup epochs
        "warmup_momentum": 0.8,   # initial momentum during warmup
        "warmup_bias_lr": 0.1,    # initial bias lr during warmup
        "box": 0.02,               # box loss gain
        "cls": 0.5,               # classification loss gain
        "iou": 0.20,              # IoU training threshold
        "hsv_h": 0.015,           # HSV hue augmentation
        "hsv_s": 0.7,             # HSV saturation augmentation
        "hsv_v": 0.4,             # HSV value augmentation
        "degrees": 0.0,           # rotation (+/- deg)
        "translate": 0.1,         # translation (+/- fraction)
        "scale": 0.9,             # scale (+/- gain)
        "shear": 0.0,             # shear (+/- deg)
        "perspective": 0.0,       # perspective (+/- fraction)
        "flipud": 0.0,            # up-down flip probability
        "fliplr": 0.5,            # left-right flip probability
        "mosaic": 1.0,            # mosaic augmentation probability
        "mixup": 0.15,            # mixup augmentation probability
        "copy_paste": 0.3,        # segment copy-paste augmentation probability
        "optimizer": "AdamW",     # optimizer
    }

    # 3) Fetch any parameter overrides from the Task config
    #    By default, these might appear under "General" or the top-level,
    #    depending on how they're logged.
    #    `task.get_parameters_as_dict()` returns a structure like:
    #    {
    #       'Args': {...},
    #       'General': {...},
    #       'Manual': {...},
    #       ...
    #    }
    #    Adjust the dict key below based on how your parameters are actually stored in the UI.
    user_params = task.get_parameters_as_dict().get("General", {})
    print("User parameters:", user_params)

    # 4) Manually override defaults with whatever is in user_params
    for key, default_val in default_params.items():
        if key in user_params:
            # Attempt to parse float -> so e.g. '2.0' can become 2
            # If that fails, keep it as string or whatever type is there
            try:
                default_params[key] = float(user_params[key])
            except ValueError:
                default_params[key] = user_params[key]
    print("Final parameters:", default_params)
    # 5) Now connect the final merged dict to ClearML (this logs them so they appear in the UI)
    params = task.connect(default_params)

    # 6) Convert numeric fields to int if YOLO expects integers
    params["epochs"] = int(float(params["epochs"]))
    params["img_size"] = int(float(params["img_size"]))
    params["batch_size"] = int(float(params["batch_size"]))

    # 7) Train with Ultralytics
    from ultralytics import YOLO
    model = YOLO(params["model_config"])
    results = model.train(
        data=params["data_config"],
        epochs=params["epochs"],
        imgsz=params["img_size"],
        batch=params["batch_size"],
        lr0=params["lr0"],
        lrf=params["lrf"],
        momentum=params["momentum"],
        weight_decay=params["weight_decay"],
        warmup_epochs=params["warmup_epochs"],
        warmup_momentum=params["warmup_momentum"],
        warmup_bias_lr=params["warmup_bias_lr"],
        box=params["box"],
        cls=params["cls"],
        iou=params["iou"],
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        degrees=params["degrees"],
        translate=params["translate"],
        scale=params["scale"],
        shear=params["shear"],
        perspective=params["perspective"],
        flipud=params["flipud"],
        fliplr=params["fliplr"],
        mosaic=params["mosaic"],
        mixup=params["mixup"],
        copy_paste=params["copy_paste"],
        optimizer=params["optimizer"],
    )
    try:
        print("logging metrics...")
        # 1) Grab the ClearML logger
        logger = task.get_logger()

        # 2) Extract metrics from YOLO results 
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            # 3) Log each metric individually to ClearML for easy comparison/plotting
            for metric_name, metric_value in results.results_dict.items():
                # If the value is numeric, log it as a scalar
                if isinstance(metric_value, (int, float)):
                    logger.report_scalar(
                        title="metrics",          # Group name in ClearML
                        series=metric_name,       # Individual metric name
                        iteration=params["epochs"],  # or results.epoch if available
                        value=metric_value
                    )

            # (Optional) Upload the entire metrics dict as an artifact (JSON-like) for later reference
            task.upload_artifact(
                name="all_metrics",
                artifact_object=results.results_dict
            )
        else:
            print("No valid metrics found in `results.results_dict`")

        print("Training completed!")
    except Exception as e:
        print(f"Error logging metrics: {e}")

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
            UniformParameterRange("General/lr0", min_value=1e-5, max_value=1e-1),
            UniformParameterRange("General/lrf", min_value=1e-3, max_value=1.0),
            UniformParameterRange("General/momentum", min_value=0.8, max_value=0.99),
            UniformParameterRange("General/weight_decay", min_value=0.0, max_value=0.001),
            UniformParameterRange("General/warmup_epochs", min_value=0.0, max_value=5.0),
            UniformParameterRange("General/warmup_momentum", min_value=0.0, max_value=0.95),
            UniformParameterRange("General/warmup_bias_lr", min_value=0.0, max_value=0.2),
            UniformParameterRange("General/box", min_value=0.02, max_value=0.2),
            UniformParameterRange("General/cls", min_value=0.2, max_value=4.0),
            UniformParameterRange("General/iou", min_value=0.1, max_value=0.8),
            UniformParameterRange("General/hsv_h", min_value=0.0, max_value=0.1),
            UniformParameterRange("General/hsv_s", min_value=0.0, max_value=0.9),
            UniformParameterRange("General/hsv_v", min_value=0.0, max_value=0.9),
            UniformParameterRange("General/degrees", min_value=0.0, max_value=45.0),
            UniformParameterRange("General/translate", min_value=0.0, max_value=0.9),
            UniformParameterRange("General/scale", min_value=0.0, max_value=0.9),
            UniformParameterRange("General/shear", min_value=0.0, max_value=10.0),
            UniformParameterRange("General/perspective", min_value=0.0, max_value=0.001),
            UniformParameterRange("General/flipud", min_value=0.0, max_value=1.0),
            UniformParameterRange("General/fliplr", min_value=0.0, max_value=1.0),
            UniformParameterRange("General/mosaic", min_value=0.0, max_value=1.0),
            UniformParameterRange("General/mixup", min_value=0.0, max_value=1.0),
            UniformParameterRange("General/copy_paste", min_value=0.0, max_value=1.0),
        ],
        # this is the objective metric we want to maximize/minimize
        objective_metric_title="metrics",
        objective_metric_series="metrics/mAP50(M)",
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
        total_max_jobs=10,
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
    best_map = best_exp.get_last_scalar_metrics().get("metrics", {}).get("metrics/mAP50(M)", {}).get("last")
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
