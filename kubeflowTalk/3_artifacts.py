from google.cloud import aiplatform as vertexai  # run my pipeline on vertex ai
from google.cloud.aiplatform import pipeline_jobs
from google.oauth2 import service_account  # authenticate to vertex ai
from kfp import compiler
from kfp.dsl import component, pipeline, Artifact, Input, Output

credentials = service_account.Credentials.from_service_account_file(
    "service-account.json"
)
vertexai.init(credentials=credentials, location="us-central1")
PIPELINE_ROOT = "gs://tamu-vertex-ai-pipelines-bucket"
PIPELINE_PACKAGING_CONFIG = "artifacts_pipeline.yaml"
PIPELINE_NAME = "artifacts_pipeline"


@component(base_image="python:3.9")
def create_file(content: str, file_artifact: Output[Artifact]):
    """
    Create a file
    """
    with open(file_artifact.path, "w") as f:
        f.write(content)

    file_artifact.metadata["name"] = "file"
    file_artifact.metadata["description"] = "A file"


@component(base_image="python:3.9")
def read_file(file_artifact: Input[Artifact]):
    """
    Read a file
    """
    with open(file_artifact.path, "r") as f:
        print(f.read())


@pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def artifacts_pipeline(file_content: str):
    """
    Pipeline definition
    """
    create_file_task = create_file(content=file_content)
    read_file_task = read_file(file_artifact=create_file_task.outputs["file_artifact"])


compiler.Compiler().compile(
    pipeline_func=artifacts_pipeline, package_path=PIPELINE_PACKAGING_CONFIG
)

job = pipeline_jobs.PipelineJob(
    display_name=PIPELINE_NAME,
    template_path=PIPELINE_PACKAGING_CONFIG,
    parameter_values={"file_content": "Hello World!!!!!!"},
)

job.run(sync=False)
