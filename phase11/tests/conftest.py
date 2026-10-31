"""Phase 11A is CPU only: the GPU belongs to the pretraining run. This hides it before torch can be imported."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
