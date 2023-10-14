from google.cloud import aiplatform as vertexai
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    filename="service-account.json"
)
vertexai.init(
    project="tamu-vertex-ai-pipelines", location="us-central1", credentials=credentials
)
models = vertexai.Model.list()
print(models)
endpoints = vertexai.Endpoint.list()
print(endpoints)


endpoint_name = "sklearn-model-endpoint"
model_name = "sklearn-model"

filtered_endpoints = vertexai.Endpoint.list(filter=f"display_name={endpoint_name}")
assert filtered_endpoints, "Endpoint not found."

filtered_models = vertexai.Model.list(filter=f"display_name={model_name}")
assert filtered_models, "model not found."


# undeploy models from endpoint and delete endpoint
endpoint: vertexai.Endpoint = filtered_endpoints[0]
for model_in_endpoint in endpoint.list_models():
    endpoint.undeploy(deployed_model_id=model_in_endpoint.id)
endpoint.delete()

# delete model
model: vertexai.Model = filtered_models[0]
model.delete()
