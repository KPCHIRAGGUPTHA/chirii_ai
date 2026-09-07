import os
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import torch

from tokenizer import CharTokenizer
from model import MiniGPT, MiniGPTConfig
from generate import generate_text, load_model_and_tokenizer
from train import train_model

# Global state for background training task
TRAIN_STATE = {
    "status": "idle", # 'idle', 'training', 'completed', 'error'
    "step": 0,
    "max_iters": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "val_perplexity": 0.0,
    "best_val_loss": 0.0,
    "elapsed_sec": 0.0,
    "history": [],
    "error": None
}

class MiniGPTRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="static", **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.path = '/index.html'
            return super().do_GET()
        elif self.path == '/api/info':
            ckpt_path = os.path.join("checkpoints", "best_model.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join("checkpoints", "checkpoint.pt")
            model_loaded = os.path.exists(ckpt_path)
            param_count = 0
            config_dict = {}
            
            if model_loaded:
                try:
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    cfg = ckpt["config"]
                    model = MiniGPT(cfg)
                    param_count = model.get_num_params()
                    config_dict = {
                        "n_layer": cfg.n_layer,
                        "n_head": cfg.n_head,
                        "n_embd": cfg.n_embd,
                        "block_size": cfg.block_size,
                        "vocab_size": cfg.vocab_size
                    }
                except Exception as e:
                    print(f"Error reading model info: {e}")

            self._send_json({
                "model_loaded": model_loaded,
                "param_count": param_count,
                "config": config_dict,
                "device": "CUDA GPU" if torch.cuda.is_available() else "CPU",
                "train_state": TRAIN_STATE
            })
        elif self.path == '/api/train/status':
            self._send_json(TRAIN_STATE)
        else:
            return super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            req = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception:
            req = {}

        if self.path == '/api/generate':
            prompt = req.get("prompt", "First Citizen:")
            max_tokens = int(req.get("max_tokens", 120))
            temperature = float(req.get("temperature", 0.8))
            top_k = int(req.get("top_k", 40)) if req.get("top_k") else None
            top_p = float(req.get("top_p", 0.9)) if req.get("top_p") else None

            try:
                generated = generate_text(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                self._send_json({
                    "success": True,
                    "prompt": prompt,
                    "generated": generated
                })
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=500)

        elif self.path == '/api/train':
            if TRAIN_STATE["status"] == "training":
                self._send_json({"success": False, "error": "Training is already in progress!"}, status=400)
                return

            iters = int(req.get("iters", 300))
            batch_size = int(req.get("batch_size", 32))
            block_size = int(req.get("block_size", 128))
            n_layer = int(req.get("n_layer", 4))
            n_embd = int(req.get("n_embd", 128))
            n_head = int(req.get("n_head", 4))

            # Custom dataset text update if provided
            custom_text = req.get("custom_text", None)
            data_path = None
            if custom_text and custom_text.strip():
                os.makedirs("data", exist_ok=True)
                data_path = os.path.join("data", "custom.txt")
                with open(data_path, "w", encoding="utf-8") as f:
                    f.write(custom_text.strip())

            # Reset train state
            TRAIN_STATE["status"] = "training"
            TRAIN_STATE["step"] = 0
            TRAIN_STATE["max_iters"] = iters
            TRAIN_STATE["train_loss"] = 0.0
            TRAIN_STATE["val_loss"] = 0.0
            TRAIN_STATE["history"] = []
            TRAIN_STATE["error"] = None

            def run_training_bg():
                def callback(stats):
                    TRAIN_STATE["step"] = stats["step"]
                    TRAIN_STATE["train_loss"] = stats["train_loss"]
                    TRAIN_STATE["val_loss"] = stats["val_loss"]
                    TRAIN_STATE["val_perplexity"] = stats.get("val_perplexity", 0.0)
                    TRAIN_STATE["best_val_loss"] = stats.get("best_val_loss", 0.0)
                    TRAIN_STATE["elapsed_sec"] = stats["elapsed_sec"]
                    TRAIN_STATE["history"].append(stats)

                try:
                    train_model(
                        data_path=data_path,
                        max_iters=iters,
                        batch_size=batch_size,
                        block_size=block_size,
                        n_layer=n_layer,
                        n_head=n_head,
                        n_embd=n_embd,
                        callback=callback
                    )
                    TRAIN_STATE["status"] = "completed"
                except Exception as e:
                    TRAIN_STATE["status"] = "error"
                    TRAIN_STATE["error"] = str(e)
                    print(f"Background training failed: {e}")

            thread = threading.Thread(target=run_training_bg, daemon=True)
            thread.start()

            self._send_json({
                "success": True,
                "message": "Training started in background thread!"
            })
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)

def run_server(port=5000):
    os.makedirs("static", exist_ok=True)
    server_address = ('', port)
    httpd = HTTPServer(server_address, MiniGPTRequestHandler)
    print(f"Mini-GPT Web Studio server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_server(port)
