"""Offline checks: python -m unittest -v test_lab.py (never contacts real cameras)."""
import queue
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import CameraInput, Lab, camera_url, create_app


class Alive:
    def is_alive(self):
        return True


class LabTests(unittest.TestCase):
    def test_camera_url_normalizes_only_http_root(self):
        self.assertEqual(camera_url('192.0.2.10:8080'), 'http://192.0.2.10:8080/video')
        self.assertEqual(camera_url('http://example.test/stream?q=1'), 'http://example.test/stream?q=1')
        self.assertEqual(camera_url('rtsp://example.test/live'), 'rtsp://example.test/live')
        for bad in ['file:///etc/passwd', 'http://', 'http://example.test:99999', 'http://x/with space', 'https://x/#part']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                camera_url(bad)

    def test_local_api_guards_and_private_validation_errors(self):
        with TestClient(create_app(lambda: Lab(background=False))) as client:
            self.assertEqual(client.get('/api/state').status_code, 200)
            self.assertEqual(client.get('/api/state', headers={'Origin': 'https://example.test'}).status_code, 403)
            self.assertEqual(client.post('/api/stop', json={}).status_code, 403)
            result = client.post('/api/camera', json={'url': 'file://user:private-token@host/x'},
                headers={'X-Vision-Lab': 'dashboard'})
            self.assertEqual(result.status_code, 422)
            self.assertNotIn('private-token', result.text)
            self.assertEqual(client.get('/api/frame.jpg').status_code, 404)
            self.assertEqual(client.get('/api/state', headers={'host': 'attacker.test'}).status_code, 400)
            self.assertEqual(client.get('/api/state').headers['cache-control'], 'no-store')

    def test_stale_frames_never_start_inference(self):
        lab = Lab(background=False)
        lab.state['running'] = True
        lab.state['camera'].update(status='streaming', frame_at=time.time()-30, frame_sequence=1)
        lab.jpeg = b'old image'
        lab.tick()
        self.assertIsNone(lab.model_process)
        self.assertEqual(lab.state['camera']['status'], 'reconnecting')
        with self.assertRaises(ValueError):
            lab.analyze()
        lab.close()

    def test_one_inference_in_flight_and_bounded_history(self):
        lab = Lab(background=False)
        lab.state['running'] = True
        lab.state['camera'].update(status='streaming', frame_at=time.time(), frame_sequence=7)
        lab.state['model']['status'] = 'ready'
        lab.jpeg = b'fixture'
        lab.model_process = Alive()
        lab.requests = queue.Queue(1)
        lab.model_output = queue.Queue()
        lab.tick()
        lab.tick()
        self.assertEqual(lab.requests.qsize(), 1)
        self.assertEqual(lab.state['model']['status'], 'analyzing')
        for index in range(35):
            lab.model_output.put({'status': 'ready', 'observation': {'id': str(index), 'text': 'fixture'}})
            lab.tick()
        self.assertEqual(len(lab.state['observations']), 30)
        self.assertEqual(lab.state['observations'][0]['id'], '34')
        lab.model_process = lab.requests = lab.model_output = None
        lab.close()

    def test_stop_detaches_old_results_and_releases_processes(self):
        lab = Lab(background=False)
        lab.state['running'] = True
        lab.jpeg = b'private frame'
        lab.job_started = 123
        lab.camera_process = Alive()
        lab.model_process = Alive()
        with patch.object(Lab, 'halt') as halt:
            lab.stop()
        self.assertEqual(halt.call_count, 2)
        self.assertIsNone(lab.jpeg)
        self.assertIsNone(lab.camera_process)
        self.assertIsNone(lab.model_process)
        self.assertEqual(lab.job_started, 0)
        self.assertFalse(lab.state['running'])

    def test_failed_spawn_can_be_stopped_and_retried(self):
        lab = Lab(background=False)
        with patch('multiprocessing.process.BaseProcess.start', side_effect=OSError('fixture')):
            lab.start(CameraInput(url='http://192.0.2.10/video'))
        self.assertFalse(lab.state['running'])
        self.assertEqual(lab.state['camera']['status'], 'error')
        lab.stop()
        self.assertIsNone(lab.camera_process)

    def test_model_timeout_detaches_queues_and_reaps(self):
        lab = Lab(background=False)
        lab.state['running'] = True
        lab.state['model']['status'] = 'analyzing'
        lab.job_started = time.monotonic() - 121
        lab.model_process = Alive()
        lab.model_output = queue.Queue()
        lab.requests = queue.Queue()
        with patch('app.threading.Thread') as reaper:
            lab.tick()
            reaper.return_value.start.assert_called_once()
        self.assertIsNone(lab.model_process)
        self.assertIsNone(lab.model_output)
        self.assertIsNone(lab.requests)
        self.assertEqual(lab.state['model']['status'], 'error')
        lab.close()


if __name__ == '__main__':
    unittest.main()
