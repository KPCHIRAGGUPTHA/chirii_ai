import sys
import os
import sys
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer import CharTokenizer
from model import MiniGPT, MiniGPTConfig
from dataset import get_or_download_text, get_batch
from generate import generate_text, load_model_and_tokenizer