import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "models/onnx/model_int8.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

MEAN = np.array([0.4815, 0.4578, 0.4082])
STD = np.array([0.2686, 0.2613, 0.2758])


def sigmoid(x):
    x = np.clip(x, -80, 80)
    return 1.0 / (1.0 + np.exp(-x))


def prepare_image(image):
    image = image.convert("RGB")

    width, height = image.size
    scale = 440 / min(width, height)

    new_size = (
        int(width * scale),
        int(height * scale)
    )

    return image.resize(new_size)


def get_crops(image):
    width, height = image.size
    size = 384

    positions = [
        (0, 0),
        (width - size, 0),
        (0, height - size),
        (width - size, height - size),
        ((width - size) // 2, (height - size) // 2)
    ]

    return [
        image.crop((x, y, x + size, y + size))
        for x, y in positions
    ]


def get_logit(image):
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD

    arr = np.expand_dims(
        arr.transpose(2, 0, 1), 0
    ).astype(np.float32)

    output = session.run(
        None,
        {input_name: arr}
    )[0]

    return float(np.asarray(output).reshape(-1)[0])


def analyze_multicrop(image):
    image = prepare_image(image)
    crops = get_crops(image)

    logits = [get_logit(crop) for crop in crops]

    probabilities = [sigmoid(x) for x in logits]

    return {
        "center_ai": float(probabilities[4]),
        "mean_ai": float(np.mean(probabilities)),
        "mean_logit_ai": float(sigmoid(np.mean(logits))),
        "max_ai": float(np.max(probabilities)),
        "crop_scores": [float(x) for x in probabilities]
    }


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        try:
            image = Image.open(path)

            result = analyze_multicrop(image)

            print("\nIMAGE:", path)
            print("Center AI:", round(result["center_ai"] * 100, 2), "%")
            print("Mean AI:", round(result["mean_ai"] * 100, 2), "%")
            print("Mean-logit AI:", round(result["mean_logit_ai"] * 100, 2), "%")
            print("Maximum crop AI:", round(result["max_ai"] * 100, 2), "%")
            print("Individual crops:", [
                round(x * 100, 2)
                for x in result["crop_scores"]
            ])

        except Exception as error:
            print("ERROR:", path, error)