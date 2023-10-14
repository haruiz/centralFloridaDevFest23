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
PIPELINE_PACKAGING_CONFIG = "process_string_pipeline.yaml"
PIPELINE_NAME = "process_string_pipeline"


@component(base_image="python:3.9")
def read_string(input_string: str) -> str:
    """
    Hello World component
    """
    return input_string


@component(base_image="python:3.9")
def transform_string(input_string: str) -> str:
    """
    Pipeline definition
    """
    output_string = input_string.upper()
    return output_string


@component(base_image="python:3.9")
def print_string(input_string: str):
    """
    Pipeline definition
    """
    print(input_string)


@pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def process_string_pipeline(input_string: str):
    """
    Pipeline definition
    """
    read_string_task = read_string(input_string=input_string)
    transform_string_task = transform_string(input_string=read_string_task.output)
    print_string_task = print_string(input_string=transform_string_task.output)


compiler.Compiler().compile(
    pipeline_func=process_string_pipeline, package_path=PIPELINE_PACKAGING_CONFIG
)
job = pipeline_jobs.PipelineJob(
    display_name=PIPELINE_NAME,
    template_path=PIPELINE_PACKAGING_CONFIG,
    parameter_values={"input_string": "Hello World from devfest central florida!"},
)

job.run(sync=False)
