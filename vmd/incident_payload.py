"""Allowlisted incident metadata shared by the local and read-only APIs."""
import json
import math
from datetime import datetime, timezone


MAX_TIMESTAMP = 253402300799


def number(value, minimum=0, maximum=MAX_TIMESTAMP):
    return (type(value) in (int, float) and minimum <= value <= maximum
            and math.isfinite(value))


def utc(value):
    if not number(value):
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def incident_record(row):
    """Never return raw signals, responder contacts, private paths or stream URLs."""
    try:
        signals = json.loads(row['signals'])
    except (TypeError, ValueError):
        raise ValueError('Incident source metadata is invalid') from None
    if not isinstance(signals, dict) or type(signals.get('live_camera')) is not bool:
        raise ValueError('Incident source metadata is invalid')
    location = signals.get('location')
    if (isinstance(location, dict) and number(location.get('latitude'), -90, 90)
            and number(location.get('longitude'), -180, 180)):
        location = {
            'latitude': location['latitude'], 'longitude': location['longitude'],
            'place': location.get('place') if isinstance(location.get('place'), str) else None,
            'source': location.get('source') if location.get('source') in ('manual', 'browser') else 'unknown',
            # Current camera configuration has no independent location verification step.
            'verified': False,
            'accuracy_meters': location.get('accuracy_meters') if number(location.get('accuracy_meters')) else None,
            'captured_at': utc(location.get('captured_at')),
        }
    else:
        location = None
    recorded = signals['live_camera'] is False
    return {
        'incident_id': row['id'], 'type': row['event_type'],
        'camera_id': row['camera_id'], 'camera_name': row['camera_name'],
        'detected_at': utc(row['created']),
        # created is detection/analysis time, not a verified occurrence clock.
        'occurred_at': None, 'source_kind': 'recording' if recorded else 'live_camera',
        'source_seconds': signals.get('source_seconds') if recorded and number(signals.get('source_seconds')) else None,
        'location': location, 'review_status': row['review'],
    }


def event_package_record(row, share):
    """Combine a saved snapshot with the evidence URL issued for that incident."""
    record = incident_record(row)
    location = json.loads(row['signals']).get('location')
    place = location.get('place') if isinstance(location, dict) else None
    return {**record, 'place': place if isinstance(place, str) else None,
            'video_url': share['url'], 'video_expires_at': utc(share['expires_at'])}
