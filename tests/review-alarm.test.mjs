import test from 'node:test';
import assert from 'node:assert/strict';
import { reviewAlarmEvents, alarmLevel } from '../vmd/static/response.mjs';

const camera = {camera_id: 'entrance', status: 'running', mode: 'live', stale: false};
const item = {id: 'item-episode', camera_id: 'entrance', mode: 'live', created: 100,
  review: 'unreviewed', handled_at: null, event_type: 'unattended_object', signals: {live_camera: true}};

test('fresh saved item/down/fall episodes are eligible for local review chimes without a dispatch alert', () => {
  for (const event_type of ['unattended_object', 'person_down', 'possible_fall']) {
    const event = {...item, event_type};
    assert.deepEqual(reviewAlarmEvents([event], [camera], 110), [event]);
    assert.equal(alarmLevel({state: 'review', event, created: 100}, 110), 0);
  }
  assert.deepEqual(reviewAlarmEvents([{...item, event_type: 'fight'}], [camera], 110), []);
  assert.deepEqual(reviewAlarmEvents(undefined, undefined, 110), []);
});

test('recordings, synthetic events, handled/reviewed events, stale cameras and old history cannot chime', () => {
  for (const change of [
    {mode: 'demo'}, {signals: {live_camera: false}}, {signals: {live_camera: 'true'}},
    {review: 'false_positive'}, {review: 'confirmed'}, {handled_at: 105},
    {created: 79}, {created: 111}, {created: NaN}, {camera_id: 'other'},
  ]) assert.deepEqual(reviewAlarmEvents([{...item, ...change}], [camera], 110), []);
  for (const change of [{status: 'finished'}, {status: 'idle'}, {stale: true}, {presentation: true}])
    assert.deepEqual(reviewAlarmEvents([item], [{...camera, ...change}], 110), []);
});
