import kfp.dsl as dsl
from google.cloud import aiplatform  # run my pipeline on vertex ai
from google.cloud.aiplatform import pipeline_jobs  # create a pipeline job on vertex ai
from google.oauth2 import service_account  # authenticate to vertex ai
from google_cloud_pipeline_components.v1.vertex_notification_email import (
    VertexNotificationEmailOp,
)  # use the vertex ai pipeline components
from kfp import compiler
from kfp.components.pipeline_task import PipelineTask
from kfp.components.task_final_status import PipelineTaskFinalStatus
from kfp.dsl import component, Input, Output, Artifact
from kfp.dsl import pipeline

credentials = service_account.Credentials.from_service_account_file(
    "service-account.json"
)
aiplatform.init(credentials=credentials, location="us-central1")
PIPELINE_ROOT = "gs://tamu-vertex-ai-pipelines-bucket"


@component(base_image="python:3.9", packages_to_install=["google-cloud-storage"])
def read_image(bucket_name: str, blob_name: str, image_artifact: Output[Artifact]):
    from google.cloud import storage
    from pathlib import Path

    output_file = Path(image_artifact.path).with_suffix(".jpg")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(output_file)

    # add metadata to the artifact
    image_artifact.metadata["name"] = "dog"
    image_artifact.metadata["description"] = "A dog"
    image_artifact.metadata["labels"] = {"task": "classification"}


@component(base_image="python:3.9", packages_to_install=["Pillow", "tqdm", "numpy"])
def resize_image(
    image_artifact: Input[Artifact],
    width: int,
    height: int,
    resized_image_artifact: Output[Artifact],
):
    from PIL import Image

    image = Image.open(image_artifact.path + ".jpg")
    image = image.resize((width, height))
    image.save(resized_image_artifact.path + ".jpg")


@component(base_image="python:3.9", packages_to_install=["Pillow", "tqdm", "numpy"])
def image2hsv(
    resized_image_artifact: Input[Artifact],
    hsv_image_artifact: Output[Artifact],
):
    from PIL import Image
    import numpy as np

    image = Image.open(resized_image_artifact.path + ".jpg")
    image = image.convert("HSV")
    image = np.asarray(image)
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image.save(hsv_image_artifact.path + ".jpg")


@component(
    base_image="python:3.9", packages_to_install=["google_cloud_pipeline_components"]
)
def exit_op(status: PipelineTaskFinalStatus):
    """Prints pipeline run status."""
    print("Pipeline status: ", status.state)
    print("Job resource name: ", status.pipeline_job_resource_name)
    print("Pipeline task name: ", status.pipeline_task_name)
    print("Error code: ", status.error_code)
    print("Error message: ", status.error_message)

    # send task status to cloud function


@pipeline(name="image-pipeline", pipeline_root=PIPELINE_ROOT)
def image_pipeline(bucket_name: str, blob_name: str, width: int, height: int):
    # print_status_task = exit_op()
    # with dsl.ExitHandler(exit_task=print_status_task, name="exit-handler"):
    notify_email_task = VertexNotificationEmailOp(recipients=["henryruiz22@gmail.com"])

    with dsl.ExitHandler(notify_email_task):
        readImageTask: PipelineTask = read_image(
            bucket_name=bucket_name, blob_name=blob_name
        )
        # readImageTask.set_memory_limit("2G")
        resizeImageTask = resize_image(
            image_artifact=readImageTask.outputs["image_artifact"],
            width=width,
            height=height,
        )

        image2hsvTask = image2hsv(
            resized_image_artifact=resizeImageTask.outputs["resized_image_artifact"]
        )


compiler.Compiler().compile(
    pipeline_func=image_pipeline, package_path="image_pipeline.json"
)

job = pipeline_jobs.PipelineJob(
    display_name="image-pipeline",
    template_path="image_pipeline.json",
    parameter_values={
        "bucket_name": "tamu-images",
        "blob_name": "dog.jpg",
        "width": 200,
        "height": 200,
    },
)

job.run(sync=False)
