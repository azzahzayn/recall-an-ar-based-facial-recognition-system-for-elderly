import os
from pydantic import BaseModel

class EmbedderConfig(BaseModel):
    model_path: str = ""
    # A standard public ArcFace ONNX model URL
    model_url: str = "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx"
    cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "face_ar_system")
    use_mock: bool = False
