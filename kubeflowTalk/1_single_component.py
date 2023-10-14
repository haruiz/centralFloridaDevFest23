from google.cloud import aiplatform as vertexai  # run my pipeline on vertex ai
from google.cloud.aiplatform import pipeline_jobs
from google.oauth2 import service_account  # authenticate to vertex ai
from kfp import compiler
from kfp.dsl import component
from kfp.dsl import pipeline

credentials = service_account.Credentials.from_service_account_file(
    "service-account.json"
)
vertexai.init(credentials=credentials, location="us-central1")
PIPELINE_ROOT = "gs://tamu-vertex-ai-pipelines-bucket"
PIPELINE_PACKAGING_CONFIG = "hello_world_pipeline.yaml"
PIPELINE_NAME = "hello-world-pipeline"


@component(base_image="python:3.9")
def hello_world():
    """
    Hello World component
    """
    print("Hello World")


@pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def hello_world_pipeline():
    """
    Pipeline definition
    """
    hello_world_task = hello_world()


compiler.Compiler().compile(
    pipeline_func=hello_world_pipeline, package_path=PIPELINE_PACKAGING_CONFIG
)
job = pipeline_jobs.PipelineJob(
    display_name=PIPELINE_NAME,
    template_path=PIPELINE_PACKAGING_CONFIG,
    parameter_values={},
)

job.run(sync=False)
