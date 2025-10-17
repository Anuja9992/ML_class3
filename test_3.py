from fastapi.testclient import TestClient
import app as app_module

client = TestClient(app_module.app)
sample = [5.1, 3.5, 1.4, 0.2]
resp = client.post('/predict', json={'data': sample})
print('Status code:', resp.status_code)
print('Response JSON:', resp.json())