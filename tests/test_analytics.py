import json
import os
import sqlite3
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from conftest import authenticate
from vmd.api import create_app
from vmd.storage import Store


@pytest.fixture
def local_timezone():
    previous = os.environ.get('TZ')
    os.environ['TZ'] = 'Asia/Kolkata'
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = previous
        time.tzset()


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path)
    try:
        yield value
    finally:
        value.close()


def add_incident(conn, identity, created, *, review='unreviewed', mode='live',
                 live_camera=True, event_type='fight', camera_id='entrance', camera_name='Entrance'):
    conn.execute('''INSERT INTO incidents
        (id,created,mode,score,reasons,signals,review,event_type,camera_id,camera_name)
        VALUES (?,?,?,?,?,?,?,?,?,?)''', (identity, created, mode, .95, '[]',
        json.dumps({'live_camera': live_camera}), review, event_type, camera_id, camera_name))


def test_summary_counts_all_records_and_uses_local_calendar_windows(store, local_timezone):
    now = datetime(2026, 9, 26, 12, 30).timestamp()
    midnight = datetime(2026, 9, 26).timestamp()
    start = datetime(2026, 9, 20).timestamp()
    month_start = datetime(2026, 8, 28).timestamp()
    with store.connect() as conn:
        for index in range(120):
            add_incident(conn, f'current-{index:03}', midnight + index,
                         review='confirmed' if index < 30 else 'unreviewed',
                         event_type='knife_detected' if index < 20 else 'fight')
        add_incident(conn, 'at-week-start', start, camera_name='Old entrance')
        add_incident(conn, 'before-week', start - 1)
        add_incident(conn, 'at-month-start', month_start)
        add_incident(conn, 'before-month', month_start - 1)
        add_incident(conn, 'unknown-type', now - 2, event_type='future_event',
                     camera_id='courtyard', camera_name='Courtyard')
        add_incident(conn, 'latest-camera-name', now - 1, event_type='gun_detected', camera_name='Main entrance')
        add_incident(conn, 'false-positive', now - 1, review='false_positive', event_type='gun_detected')
        add_incident(conn, 'demo', now - 1, mode='demo')
        add_incident(conn, 'recording', now - 1, live_camera=False)
        add_incident(conn, 'unknown-source', now - 1, live_camera=None)
        add_incident(conn, 'not-a-boolean', now - 1, live_camera=1)
        add_incident(conn, 'future', now + 1)
        add_incident(conn, 'malformed-signals', now - 1)
        conn.execute("UPDATE incidents SET signals='broken' WHERE id='malformed-signals'")
    result = store.analytics_summary(7, now=now)
    assert result['start'] == start and result['generated_at'] == now and result['days'] == 7
    assert result['timezone'] == 'IST'
    assert result['total'] == 123 and result['today'] == 122  # Not the library's latest-100 limit.
    assert result['weapon'] == 21 and result['confirmed'] == 30 and result['unreviewed'] == 93
    assert result['false_positives'] == 1
    assert result['trend'] == [
        {'date': '2026-09-20', 'value': 1},
        *[{'date': f'2026-09-{day}', 'value': 0} for day in range(21, 26)],
        {'date': '2026-09-26', 'value': 122},
    ]
    assert sum(item['value'] for item in result['by_type']) == result['total']
    assert {'event_type': 'future_event', 'value': 1} in result['by_type']
    assert result['by_camera'] == [
        {'camera_id': 'entrance', 'camera_name': 'Main entrance', 'value': 122},
        {'camera_id': 'courtyard', 'camera_name': 'Courtyard', 'value': 1},
    ]
    month = store.analytics_summary(30, now=now)
    assert month['total'] == 125 and month['start'] == month_start and len(month['trend']) == 30
    assert month['trend'][0] == {'date': '2026-08-28', 'value': 1}
    assert store.review('latest-camera-name', 'false_positive')
    reviewed = store.analytics_summary(7, now=now)
    assert reviewed['total'] == 122 and reviewed['weapon'] == 20 and reviewed['false_positives'] == 2


def test_empty_summary_has_measured_zeroes_without_filler(store, local_timezone):
    now = datetime(2026, 9, 26, 12, 30).timestamp()
    result = store.analytics_summary(7, now=now)
    assert all(result[key] == 0 for key in ('total', 'today', 'weapon', 'confirmed', 'unreviewed', 'false_positives'))
    assert len(result['trend']) == 7 and all(row['value'] == 0 for row in result['trend'])
    assert result['by_type'] == result['by_camera'] == result['crowd'] == []
    assert result['crowd_since'] == now - 86400


def test_calendar_boundary_across_dst_uses_local_midnight(store, local_timezone):
    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    now = datetime(2026, 11, 3, 12).timestamp()
    start = datetime(2026, 10, 28).timestamp()
    with store.connect() as conn:
        add_incident(conn, 'included', start)
        add_incident(conn, 'excluded', start - 1)
    result = store.analytics_summary(7, now=now)
    assert result['start'] == start and result['total'] == 1
    assert result['trend'][0] == {'date': '2026-10-28', 'value': 1}
    assert result['timezone'] == 'EST'


def test_telemetry_provenance_migration_and_legacy_writes_remain_compatible(tmp_path, local_timezone):
    now = datetime(2026, 9, 26, 12, 30).timestamp()
    with sqlite3.connect(tmp_path / 'telemetry.sqlite3') as conn:
        conn.execute('CREATE TABLE telemetry (created REAL, people INTEGER, score REAL, mode TEXT)')
        conn.execute("INSERT INTO telemetry VALUES (?, 90, .1, 'live')", (now - 1,))
    value = Store(tmp_path)
    try:
        for payload in (
            (now - 1, 90, .1, 'live'),
            (now - 1, 90, .1, 'live', 'legacy'),
            (now - 60, 6, .1, 'live', 'entrance', True),
            (now - 120, 8, .1, 'live', 'entrance', True),
            (now - 1, 90, .1, 'live', 'recording', False),
            (now - 1, 90, .1, 'demo', 'demo', True),
            (now + 1, 90, .1, 'live', 'future', True),
            (now - 86401, 90, .1, 'live', 'old', True),
        ):
            value.enqueue('telemetry', payload)
        value.jobs.join()
        assert value.error is None
        with value.connect() as conn:
            assert conn.execute('SELECT COUNT(*) FROM telemetry WHERE live_camera IS NULL').fetchone()[0] == 3
            assert 'incidents_time' in {row[1] for row in conn.execute('PRAGMA index_list(incidents)')}
        result = value.analytics_summary(7, now=now, camera_names={'entrance': 'Main entrance'})
        assert result['crowd'] == [{'hour': int((now - 60) // 3600) * 3600, 'average': 7,
                                    'peak': 8, 'samples': 2, 'camera_id': 'entrance',
                                    'camera_name': 'Main entrance'}]
        # The old endpoint still accepts/returns legacy rows for older clients.
        assert 'legacy' in {row['camera_id'] for row in value.analytics(now - 3600)}
    finally:
        value.close()


def test_summary_endpoint_is_authenticated_and_validates_window(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get('/api/analytics/summary').status_code == 401
        authenticate(client)
        for days in (7, 30):
            response = client.get(f'/api/analytics/summary?days={days}')
            assert response.status_code == 200
            assert response.json()['days'] == days and len(response.json()['trend']) == days
            assert response.headers['cache-control'] == 'no-store'
        assert client.get('/api/analytics/summary').json()['days'] == 7
        for days in ('0', '31', '-7', 'abc', '7.0'):
            assert client.get(f'/api/analytics/summary?days={days}').status_code == 422
        assert client.get('/api/analytics').json() == []
