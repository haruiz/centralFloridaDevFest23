from google.cloud import aiplatform as vertexai  # run my pipeline on vertex ai
from google.cloud.aiplatform import pipeline_jobs
from google.oauth2 import service_account  # authenticate to vertex ai
from kfp import compiler
from kfp.dsl import pipeline, container_component, ContainerSpec

credentials = service_account.Credentials.from_service_account_file(
    "service-account.json"
)
vertexai.init(credentials=credentials, location="us-central1")
PIPELINE_PACKAGING_CONFIG = "container_comp_pipeline.yaml"
PIPELINE_NAME = "container_comp_pipeline"


@container_component
def say_hello():
    return ContainerSpec(image="alpine", command=["echo"], args=["Hello"])


@pipeline
def hello_pipeline():
    say_hello()


compiler.Compiler().compile(
    pipeline_func=hello_pipeline, package_path=PIPELINE_PACKAGING_CONFIG
)

job = pipeline_jobs.PipelineJob(
    display_name=PIPELINE_NAME,
    template_path=PIPELINE_PACKAGING_CONFIG,
    parameter_values={},
)

job.run(sync=False)
