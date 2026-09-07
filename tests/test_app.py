import threading
import json
import time
import urllib.request
import urllib.error
from http.server import HTTPServer
import pytest
from app import MiniGPTRequestHandler

@pytest.fixture(scope="module")
def app_server():
    """Start MiniGPTRequestHandler HTTP server on a dynamic free port."""
    server = HTTPServer(('127.0.0.1', 0), MiniGPTRequestHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()

def test_api_info_endpoint(app_server):
    url = f"{app_server}/api/info"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert "model_loaded" in data
        assert "param_count" in data
        assert "device" in data
        assert "train_state" in data

def test_api_train_status_endpoint(app_server):
    url = f"{app_server}/api/train/status"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert "status" in data
        assert "step" in data

def test_api_invalid_endpoint(app_server):
    url = f"{app_server}/api/nonexistent"
    req = urllib.request.Request(url)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 404
