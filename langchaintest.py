from langchain_google_vertexai import ChatVertexAI
import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "AIzaSyAeToJ4AeUDvZHpa5ir705biWtbT_vbYVE"

# Set the path to your service account JSON file
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")

model = ChatVertexAI(model_name="gemini-2.0-flash-001", project="gen-lang-client-0391653967", key="AIzaSyAeToJ4AeUDvZHpa5ir705biWtbT_vbYVE")
print(model.invoke("Hello, world!"))