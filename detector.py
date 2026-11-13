import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "models/onnx/model_int8.onnx"

def analyze_image(image):
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    # Preserve aspect ratio; resize shortest edge to 440
    image = image.convert("RGB")
    width, height = image.size

    scale = 440 / min(width, height)
    new_size = (
        int(width * scale),
        int(height * scale)
    )

    image = image.resize(new_size)

    # Center crop to 384 x 384
    left = (image.width - 384) // 2
    top = (image.height - 384) // 2

    image = image.crop(
        (left, top, left + 384, top + 384)
    )

    # CLIP normalization
    arr = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.array([0.4815, 0.4578, 0.4082])
    std = np.array([0.2686, 0.2613, 0.2758])

    arr = (arr - mean) / std
    arr = np.expand_dims(arr.transpose(2, 0, 1), 0)

    # AI inference
    input_name = session.get_inputs()[0].name
    output = session.run(
        None,
        {input_name: arr.astype(np.float32)}
    )[0]

    logit = float(np.asarray(output).reshape(-1)[0])

    fake_probability = 1.0 / (1.0 + np.exp(-logit))
    print("RAW MODEL OUTPUT:", output)
    print("MODEL LOGIT:", logit)
    print("FAKE PROBABILITY:", fake_probability)

    return fake_probability
