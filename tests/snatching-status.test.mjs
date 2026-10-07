import test from 'node:test';
import assert from 'node:assert/strict';
import {snatchingStatusText} from '../vmd/static/snatching-status.mjs';

test('snatching status survives a fight winning the current display', () => {
  const camera = {status: 'running', assessment: 'fight_detected', object_meta: {status: 'ready'},
    settings: {snatching_vehicles: true}, signals: {
      snatching: {phase: 'waiting_reach', blockers: ['Needs a supported wrist reach']},
      vehicle_snatching: {phase: 'waiting_reaction', candidates: [{phase: 'waiting_reaction', blockers: ['Needs abrupt movement of the other person']}]}
    }};
  const status = snatchingStatusText(camera);
  assert.match(status, /On foot: waiting reach/);
  assert.match(status, /Rider review: waiting reaction/);
  assert.match(status, /abrupt movement/);
});

test('paused and stale cameras do not display a current snatching stage', () => {
  for (const options of [{status:'finished'}, {status:'running', stale:true}]) {
    const status = snatchingStatusText({...options, settings:{snatching_vehicles:true}, signals:{snatching:{phase:'confirmed'}}});
    assert.equal(status, 'Snatching check paused · rider review enabled');
  }
});

test('rider context errors and missing samples are explicit', () => {
  const camera = {status:'running', settings:{snatching_vehicles:true}};
  assert.match(snatchingStatusText(camera), /fresh vehicle sample/);
  assert.match(snatchingStatusText({...camera,object_meta:{status:'error'}}), /check the object model/);
  assert.match(snatchingStatusText({status:'running', settings:{snatching_vehicles:false}}), /enable in camera settings/);
});
