"""Recorded analysis samples every chosen frame despite startup/inference delay."""
import time

import cv2
import numpy as np
import pytest

from vmd.capture import Capture


class Video:
    def __init__(self, fps=30, count=30, shape=(720, 1280)):
        self.fps, self.count, self.shape = fps, count, shape
        self.position, self.reads, self.grabs = 0, 0, 0
        self.released = False

    def isOpened(self): return True

    def get(self, key):
        return self.fps if key == cv2.CAP_PROP_FPS else self.count if key == cv2.CAP_PROP_FRAME_COUNT else 0

    def read(self):
        if self.position >= self.count:
            return False, None
        image = np.full((*self.shape, 3), self.position, np.uint8)
        self.position += 1
        self.reads += 1
        return True, image

    def grab(self):
        if self.position >= self.count:
            return False
        self.position += 1
        self.grabs += 1
        return True

    def release(self): self.released = True


def recording(monkeypatch, tmp_path, video, target_fps=8):
    path = tmp_path/'recording.avi'
    path.touch()
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: video)
    capture = Capture(str(path), target_fps=target_fps)
    capture.start()
    return capture


def wait_packet(capture, after=0):
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        packet = capture.latest(after)
        if packet is not None:
            return packet
        if capture.finished:
            break
        time.sleep(.002)
    return None


def test_first_packet_waits_for_startup_and_explicit_consumption(monkeypatch, tmp_path):
    video = Video()
    capture = recording(monkeypatch, tmp_path, video)
    try:
        first = wait_packet(capture)
        time.sleep(.12)  # Models can take arbitrary time to warm up.
        assert capture.latest(0) is first
        assert first[0:2] == (1, 0)
        assert video.reads == 1 and video.grabs == 0
        assert not capture.finished
        second = wait_packet(capture, first[0])
        assert second[0] == 2 and second[1] == pytest.approx(4/30)
        time.sleep(.06)  # Slow inference does not silently advance the recording.
        assert capture.latest(first[0]) is second
        assert video.reads == 2
    finally:
        capture.stop()


@pytest.mark.parametrize('fps,target,count,indices', [
    (30, 8, 30, [0, 4, 8, 12, 15, 19, 23, 27]),
    (20, 8, 6, [0, 3, 5]),
    (5, 8, 5, [0, 1, 2, 3, 4]),
])
def test_file_sampling_is_deterministic_and_preserves_source_timestamps(monkeypatch, tmp_path, fps, target, count, indices):
    video = Video(fps=fps, count=count, shape=(80, 120))
    capture = recording(monkeypatch, tmp_path, video, target)
    packets = []
    try:
        after = 0
        while (packet := wait_packet(capture, after)) is not None:
            packets.append(packet)
            after = packet[0]
            time.sleep(.005)
        assert capture.finished and capture.completed and capture.error is None
        assert [int(packet[2][0, 0, 0]) for packet in packets] == indices
        assert [packet[1] for packet in packets] == pytest.approx([index/fps for index in indices])
        assert [packet[0] for packet in packets] == list(range(1, len(indices)+1))
        assert video.reads == len(indices) and video.grabs == count-len(indices)
        assert capture.metadata()['duration_seconds'] == pytest.approx(count/fps)
        assert capture.metadata()['progress'] == 1
    finally:
        capture.stop()
    assert video.released


@pytest.mark.parametrize('shape,expected', [((1200, 800), (480, 320)), ((720, 2048), (336, 960)), ((241, 321), (240, 320))])
def test_recording_resize_keeps_bounded_even_aspect_dimensions(monkeypatch, tmp_path, shape, expected):
    capture = recording(monkeypatch, tmp_path, Video(count=1, shape=shape))
    try:
        frame = wait_packet(capture)[2]
        assert frame.shape[:2] == expected
        assert frame.shape[0] <= 480 and frame.shape[1] <= 960
        assert abs(frame.shape[1]/frame.shape[0] - shape[1]/shape[0]) < .02
    finally:
        capture.stop()


@pytest.mark.parametrize('direct_stop', [False, True])
def test_stopping_without_consumer_releases_reader_without_false_completion(monkeypatch, tmp_path, direct_stop):
    video = Video(count=100)
    capture = recording(monkeypatch, tmp_path, video)
    assert wait_packet(capture)[0] == 1
    start = time.monotonic()
    if direct_stop:
        capture.stop_event.set()
        capture.thread.join(timeout=1)
    else:
        capture.stop()
    assert time.monotonic()-start < 1
    assert not capture.thread.is_alive()
    assert video.released and capture.finished and not capture.completed
    assert capture.metadata()['progress'] == 0


def test_live_capture_still_publishes_latest_frame_without_backpressure(monkeypatch):
    video = Video(count=30, shape=(1200, 800))
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *args: video)
    capture = Capture('0', target_fps=2)
    capture.start()
    capture.thread.join(timeout=2)
    try:
        packet = capture.latest(0)
        assert packet[0] == 30 and video.reads == 30 and video.grabs == 0
        assert packet[2].shape[:2] == (1200, 800)  # Live resize behavior is unchanged.
        assert capture.error and 'disconnected' in capture.error
    finally:
        capture.stop()


def test_real_video_decoder_grab_skips_only_requested_sampling_frames(tmp_path):
    path = tmp_path/'sampled.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (96, 64))
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((64, 96, 3), index*7, np.uint8))
    writer.release()
    capture = Capture(str(path), target_fps=8)
    capture.start()
    packets = []
    try:
        after = 0
        while (packet := wait_packet(capture, after)) is not None:
            packets.append(packet)
            after = packet[0]
        indices = [0, 4, 8, 12, 15, 19, 23, 27]
        assert len(packets) == len(indices)
        assert [packet[1] for packet in packets] == pytest.approx([index/30 for index in indices])
        assert [float(packet[2].mean()) for packet in packets] == pytest.approx([index*7 for index in indices], abs=1)
        assert capture.completed and capture.error is None
    finally:
        capture.stop()


@pytest.mark.parametrize('rate', [0, -1, float('inf'), float('nan')])
def test_invalid_capture_sampling_rate_is_rejected(rate):
    with pytest.raises(ValueError, match='frame rate'):
        Capture('0', target_fps=rate)
