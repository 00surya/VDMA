import json

import pytest

from vmd.incident_payload import event_package_record, incident_record, utc


def event(*, live=True, location=None, source_seconds=12.5, created=1700000000):
    return {
        'id': 'event-1', 'event_type': 'fight', 'created': created,
        'camera_id': 'camera-1', 'camera_name': 'Fictional entrance', 'review': 'unreviewed',
        'clip': '/private/evidence.avi', 'contacts': ['private contact'],
        'signals': json.dumps({'live_camera': live, 'source_seconds': source_seconds,
                              'location': location, 'private_stream': 'rtsp://secret:password@private.invalid'}),
    }


SHARE = {'incident_id': 'event-1', 'url': 'https://evidence.example/e/fictional',
         'expires_at': 1700003600, 'private_extra': 'do not include'}


def test_package_allowlist_saved_location_and_utc_clocks():
    location = {'latitude': 28.6, 'longitude': 77.2, 'place': 'Fictional north entrance',
                'source': 'browser', 'verified': True, 'accuracy_meters': 20,
                'captured_at': 1699999900, 'private_location_notes': 'hidden'}
    record = event_package_record(event(location=location), SHARE)
    assert record == {
        'incident_id': 'event-1', 'type': 'fight', 'camera_id': 'camera-1',
        'camera_name': 'Fictional entrance', 'review_status': 'unreviewed',
        'detected_at': '2023-11-14T22:13:20.000Z', 'occurred_at': None,
        'source_kind': 'live_camera', 'source_seconds': None,
        'location': {'latitude': 28.6, 'longitude': 77.2, 'place': 'Fictional north entrance',
                     'source': 'browser', 'verified': False, 'accuracy_meters': 20,
                     'captured_at': '2023-11-14T22:11:40.000Z'},
        'place': 'Fictional north entrance', 'video_url': SHARE['url'],
        'video_expires_at': '2023-11-14T23:13:20.000Z',
    }
    serialized = json.dumps(record, allow_nan=False)
    assert not any(value in serialized for value in ('private', 'password', 'rtsp', 'avi', 'hidden'))


def test_locationless_recording_preserves_analysis_time_and_video_offset():
    record = event_package_record(event(live=False), SHARE)
    assert record['source_kind'] == 'recording' and record['source_seconds'] == 12.5
    assert record['detected_at'] == '2023-11-14T22:13:20.000Z'
    assert record['occurred_at'] is None
    assert record['location'] is record['place'] is None


@pytest.mark.parametrize('location', [None, [], 'unknown', {'place': 'Fictional gate'},
    {'latitude': True, 'longitude': 2}, {'latitude': 91, 'longitude': 2},
    {'latitude': 1, 'longitude': -181}, {'latitude': float('nan'), 'longitude': 2},
    {'latitude': 1, 'longitude': float('inf')}, {'latitude': '28.6', 'longitude': 2}])
def test_invalid_coordinates_are_null_but_a_saved_place_is_retained(location):
    record = event_package_record(event(location=location), SHARE)
    assert record['location'] is None
    assert record['place'] == ('Fictional gate' if isinstance(location, dict) and 'place' in location else None)
    json.dumps(record, allow_nan=False)


def test_zero_coordinates_remain_valid_without_inventing_location_provenance():
    record = event_package_record(event(location={
        'latitude': 0, 'longitude': 0, 'place': [], 'source': 'untrusted',
        'accuracy_meters': float('nan'), 'captured_at': float('inf'),
    }), SHARE)
    assert record['location'] == {'latitude': 0, 'longitude': 0, 'place': None,
                                   'source': 'unknown', 'verified': False,
                                   'accuracy_meters': None, 'captured_at': None}
    assert record['place'] is None
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('value', [None, True, -1, '1700000000', float('nan'), float('inf'), 253402300800])
def test_invalid_numbers_are_never_exported_as_clocks_or_recording_offsets(value):
    record = event_package_record(event(live=False, source_seconds=value, created=value), SHARE)
    assert record['source_seconds'] is record['detected_at'] is utc(value) is None
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize('signals', ['broken', 'null', '[]', '{}', '{"live_camera": 1}',
                                    '{"live_camera": "false"}'])
def test_malformed_or_unknown_source_metadata_is_rejected(signals):
    row = {**event(), 'signals': signals}
    with pytest.raises(ValueError, match='source metadata is invalid'):
        incident_record(row)
    with pytest.raises(ValueError, match='source metadata is invalid'):
        event_package_record(row, SHARE)
