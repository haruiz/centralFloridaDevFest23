from typing import List

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from google.oauth2 import service_account
from kfp import compiler
from kfp import dsl
from kfp.dsl import (
    pipeline,
    component,
    Dataset,
    Output,
    Input,
    Model,
    Metrics,
    ClassificationMetrics,
    PipelineTaskFinalStatus,
    Artifact,
)

PIPELINE_ROOT = "gs://tamu-vertex-ai-pipelines-bucket"
PIPELINE_CONFIG_PATH = "ml_pipeline.yaml"
credentials = service_account.Credentials.from_service_account_file(
    "service-account.json"
)
aiplatform.init(credentials=credentials, location="us-central1")


@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas",
        "google-cloud-storage",
    ],
)
def grab_data(bucket_name: str, blob_name: str, dataset_artifact: Output[Dataset]):
    """
    Grab data from GCS and save it as a dataset
    @param bucket_name: The name of the bucket
    @param blob_name: The name of the blob
    @param dataset: The output dataset
    @return: None
    """
    from google.cloud import storage
    from pathlib import Path

    # download the dataset and save it as a csv
    dataset_path = Path(dataset_artifact.path).with_suffix(".csv")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dataset_path)

    # create a dataset from the csv
    dataset_artifact.metadata["name"] = "iris-data"
    dataset_artifact.metadata["description"] = "Iris dataset"
    dataset_artifact.metadata["labels"] = {"task": "classification"}


@component(packages_to_install=["pandas"])
def log_statistics(dataset_artifact: Input[Dataset]):
    """Log statistics about the dataset"""
    import pandas as pd
    from pathlib import Path

    dataset_path = Path(dataset_artifact.path).with_suffix(".csv")
    df = pd.read_csv(dataset_path)
    print(df.columns)
    print(df.describe())


@component(packages_to_install=["pandas", "scikit-learn"])
def split_dataset(
    test_size: float,
    dataset_artifact: Input[Dataset],
    train_dataset_artifact: Output[Dataset],
    test_dataset_artifact: Output[Dataset],
):
    """Split the dataset into train and test"""
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(Path(dataset_artifact.path).with_suffix(".csv"))
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["variety"]
    )

    train_dataset_path = Path(train_dataset_artifact.path).with_suffix(".csv")
    test_dataset_path = Path(test_dataset_artifact.path).with_suffix(".csv")

    train_dataset_artifact.metadata["name"] = "iris-train-data"
    train_dataset_artifact.metadata["size"] = len(train_df)
    train_df.to_csv(train_dataset_path, index=False)

    test_dataset_artifact.metadata["name"] = "iris-test-data"
    test_dataset_artifact.metadata["size"] = len(test_df)
    test_df.to_csv(test_dataset_path, index=False)


@component(packages_to_install=["pandas", "scikit-learn"])
def score_model(
    algorithm: str, dataset_artifact: Input[Dataset], model_artifact: Output[Model]
) -> float:
    """Train a model on the dataset"""
    import pandas as pd
    from pathlib import Path
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from joblib import dump

    algorithms_dict = {
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=1),
        "Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(kernel="linear"),
        "Logistic Regression": LogisticRegression(),
    }

    # Train the models
    dataset_path = Path(dataset_artifact.path).with_suffix(".csv")

    df = pd.read_csv(dataset_path)
    X = df.drop("variety", axis=1)
    y = df["variety"]

    # get the accuracy score
    cls = algorithms_dict[algorithm]
    accuracy_scores = cross_val_score(cls, X, y, cv=5)
    accuracy = accuracy_scores.mean()

    # update the model artifact metadata
    model_artifact.metadata["algorithm"] = algorithm
    model_artifact.metadata["score"] = accuracy

    # Save the model to the model artifact
    dump(cls, model_artifact.path)
    return accuracy


@component(packages_to_install=["pandas", "scikit-learn"])
def train_best_model(
    score_models: Input[List[Model]],
    train_dataset_artifact: Input[Dataset],
    best_model_artifact: Output[Model],
):
    from pathlib import Path
    import pandas as pd
    from joblib import load, dump

    # read dataset
    train_dataset_path = Path(train_dataset_artifact.path).with_suffix(".csv")
    df = pd.read_csv(train_dataset_path)
    X = df.drop("variety", axis=1)
    y = df["variety"]

    # get the best model
    best_scored_model = max(score_models, key=lambda model: model.metadata["score"])

    # train the best model
    cls = load(best_scored_model.path)
    cls.fit(X, y)

    # update the best model artifact metadata and save the model
    best_model_artifact.metadata = best_scored_model.metadata
    best_model_path = Path(best_model_artifact.path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(cls, best_model_path.parent / "model.joblib")


@component(packages_to_install=["pandas", "scikit-learn"])
def generate_best_model_metrics(
    best_model_artifact: Input[Model],
    test_dataset_artifact: Input[Dataset],
    metrics_artifact: Output[Metrics],
    classification_metrics_artifact: Output[ClassificationMetrics],
) -> float:
    """Generate metrics for the model"""
    from pathlib import Path
    from sklearn import metrics
    from joblib import load
    import pandas as pd

    model_path = Path(best_model_artifact.path).parent / "model.joblib"
    dataset_path = Path(test_dataset_artifact.path).with_suffix(".csv")

    df = pd.read_csv(dataset_path)
    X = df.drop("variety", axis=1)
    y = df["variety"]

    model = load(model_path)
    y_pred = model.predict(X)

    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average="macro")
    recall = metrics.recall_score(y, y_pred, average="macro")
    f1 = metrics.f1_score(y, y_pred, average="macro")

    metrics_artifact.log_metric("accuracy", accuracy)
    metrics_artifact.log_metric("precision", precision)
    metrics_artifact.log_metric("recall", recall)
    metrics_artifact.log_metric("f1", f1)

    classification_metrics_artifact.log_confusion_matrix(
        categories=["setosa", "versicolor", "virginica"],
        matrix=metrics.confusion_matrix(y, y_pred).tolist(),
    )
    return accuracy


@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform", "scikit-learn"],
)
def deploy_model(region: str, best_model_artifact: Input[Artifact]):
    """Deploy the model"""
    from google.cloud import aiplatform
    from pathlib import Path

    SKLEARN_MODEL_URI = str(Path(best_model_artifact.path).parent).replace(
        "/gcs/", "gs://"
    )
    MODEL_NAME = "sklearn-model"
    ENDPOINT_NAME = "sklearn-model-endpoint"
    GCP_PROJECT = "tamu-vertex-ai-pipelines"
    SERVING_CONTAINER_IMAGE_URI = (
        "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest"
    )
    MACHINE_TYPE = "n1-standard-4"

    aiplatform.init(project=GCP_PROJECT, location=region)
    models = aiplatform.Model.list(filter=(f"display_name={MODEL_NAME}"))
    if len(models) > 0:
        print("Model already deployed")
        model = aiplatform.Model.upload(
            parent_model=models[0].resource_name,
            display_name=MODEL_NAME,
            artifact_uri=SKLEARN_MODEL_URI,
            serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
        )
    else:
        print("Model not deployed")
        model = aiplatform.Model.upload(
            display_name=MODEL_NAME,
            artifact_uri=SKLEARN_MODEL_URI,
            serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
        )
    model.wait()

    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
    endpoint = model.deploy(
        endpoint=endpoint,
        machine_type=MACHINE_TYPE,
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
    )

    # check if model is already deployed and undeploy
    # for model in endpoint.list_models():
    #     if model.id not in endpoint.traffic_split:
    #         endpoint.undeploy(deployed_model_id=model.id)

    # test model
    X_test = [[0.5, 0.5, 0.5, 0.5]]
    print(endpoint.predict(instances=X_test).predictions)


@dsl.component
def exit_op(status: PipelineTaskFinalStatus):
    """Prints pipeline run status."""
    print("Pipeline status: ", status.state)
    print("Job resource name: ", status.pipeline_job_resource_name)
    print("Pipeline task name: ", status.pipeline_task_name)
    print("Error code: ", status.error_code)
    print("Error message: ", status.error_message)

    # send task status to cloud function


@pipeline(name="ml-pipeline", pipeline_root=PIPELINE_ROOT)
def my_pipeline(bucket_name: str, blob_name: str):
    print_status_task = exit_op()
    with dsl.ExitHandler(exit_task=print_status_task, name="exit-handler"):
        grab_data_op = grab_data(bucket_name=bucket_name, blob_name=blob_name)
        dataset_artifact = grab_data_op.outputs["dataset_artifact"]

        split_dataset_op = split_dataset(
            dataset_artifact=dataset_artifact, test_size=0.2
        )
        training_dataset_artifact = split_dataset_op.outputs["train_dataset_artifact"]
        test_dataset_artifact = split_dataset_op.outputs["test_dataset_artifact"]

        log_statistics_op = log_statistics(dataset_artifact=dataset_artifact)

        with dsl.ParallelFor(
            items=[
                "Decision Tree",
                "Naive Bayes",
                "LDA",
                "QDA",
                "KNN",
                "SVM",
                "Logistic Regression",
            ],
            name="score-models",
        ) as algorithm_name:
            score_model_task = score_model(
                algorithm=algorithm_name, dataset_artifact=dataset_artifact
            )

        score_models_artifacts = dsl.Collected(
            score_model_task.outputs["model_artifact"]
        )
        train_best_model_op = train_best_model(
            score_models=score_models_artifacts,
            train_dataset_artifact=training_dataset_artifact,
        )
        best_model_artifact = train_best_model_op.outputs["best_model_artifact"]
        generate_metrics_op = generate_best_model_metrics(
            best_model_artifact=best_model_artifact,
            test_dataset_artifact=test_dataset_artifact,
        )

        with dsl.Condition(
            generate_metrics_op.outputs["Output"] > 0.7, name="check-accuracy"
        ):
            deploy_model_op = deploy_model(
                region="us-central1", best_model_artifact=best_model_artifact
            )


compiler.Compiler().compile(
    pipeline_func=my_pipeline, package_path=PIPELINE_CONFIG_PATH
)

job = pipeline_jobs.PipelineJob(
    display_name="ml2-pipeline",
    template_path=PIPELINE_CONFIG_PATH,
    parameter_values={
        "bucket_name": "tamu-vertex-ai-pipelines-data",
        "blob_name": "iris-data.csv",
    },
)

job.run(sync=False)
