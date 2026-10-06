import test from 'node:test';
import assert from 'node:assert/strict';
import { timecode, seekTime } from '../vmd/static/player.mjs';

test('recording seek controls clamp to the video bounds and retain fractional time', () => {
  assert.equal(seekTime(3, -10, 60), 0);
  assert.equal(seekTime(57, 10, 60), 60);
  assert.equal(seekTime(13.25, -10, 60), 3.25);
  assert.equal(seekTime(0, 10, NaN), 0);
  assert.equal(timecode(3661.4), '01:01:01');
  assert.equal(timecode(NaN), '00:00:00');
});
