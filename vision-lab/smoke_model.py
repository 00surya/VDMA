"""Real GPU smoke test on generated shapes; no private footage, no accuracy claim."""
import json
import os
from io import BytesIO
from workers import configure_cache, MODEL_DIR, describe

if __name__ == '__main__':
    configure_cache()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from mlx_vlm import load
    from PIL import Image, ImageDraw
    model, processor = load(str(MODEL_DIR), trust_remote_code=False)
    image = Image.new('RGB', (640, 360), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 60, 270, 260), fill='red')
    draw.ellipse((340, 80, 550, 290), fill='blue')
    stream = BytesIO()
    image.save(stream, format='JPEG')
    for _ in range(2):
        text, metrics = describe(model, processor, stream.getvalue())
        assert text and metrics['latency_ms'] > 0
        print(json.dumps({'text': text, **metrics}), flush=True)
