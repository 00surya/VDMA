"""Optional synthetic IP camera for browser QA: python test_camera.py [port]."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import sys
import time
from PIL import Image, ImageDraw


class Camera(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != '/video':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                image = Image.new('RGB', (640, 360), '#eef2f5')
                draw = ImageDraw.Draw(image)
                x = 50 + int(time.time() * 15) % 70
                draw.rectangle((x, 75, x+180, 260), fill='#e43341')
                draw.ellipse((360, 80, 555, 275), fill='#156de4')
                draw.text((18, 18), 'SYNTHETIC CAMERA TEST - NO REAL FOOTAGE', fill='black')
                stream = BytesIO()
                image.save(stream, format='JPEG')
                jpeg = stream.getvalue()
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                    + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                self.wfile.flush()
                time.sleep(.1)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8771
    print(f'Synthetic camera: http://127.0.0.1:{port}/video', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Camera).serve_forever()
