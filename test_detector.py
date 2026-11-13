import sys
from PIL import Image
from detector import analyze_image

if len(sys.argv) < 2:
    print("Usage: python test_detector.py image1.jpg image2.png")
    sys.exit(1)

for path in sys.argv[1:]:
    try:
        image = Image.open(path).convert("RGB")
        ai_score = analyze_image(image)
        real_score = 1.0 - ai_score

        print("\n------------------------------")
        print("Image:", path)
        print(f"AI Score:   {ai_score * 100:.2f}%")
        print(f"Real Score: {real_score * 100:.2f}%")
        print("------------------------------")

    except Exception as e:
        print(f"Error processing {path}: {e}")