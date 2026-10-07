import test from 'node:test';
import assert from 'node:assert/strict';
import {overlaysEnabled, displayFramesPath, renderOverlayButton} from '../vmd/static/overlay-display.mjs';

test('overlays default on and remember only an explicit false preference', () => {
  for (const preference of [undefined, {}, {showDetectionOverlays:true}, {showDetectionOverlays:'false'}])
    assert.equal(overlaysEnabled(preference), true);
  assert.equal(overlaysEnabled({showDetectionOverlays:false}), false);
});

test('hidden previews change only the read-only frame query', () => {
  assert.equal(displayFramesPath('/camera-frames', true), '/camera-frames');
  assert.equal(displayFramesPath('/cameras/fixture/frames', false), '/cameras/fixture/frames?overlays=false');
  assert.equal(displayFramesPath('/camera-frames?view=grid', false), '/camera-frames?view=grid&overlays=false');
});

test('grid and feed controls expose the same accessible display state', () => {
  for (const enabled of [true,false]) {
    const attributes = {}, button = {ownerDocument:{createTextNode:text=>({text})},
      replaceChildren(...children){this.children=children;}, setAttribute(name,value){attributes[name]=value;}};
    renderOverlayButton(button, enabled, name=>({icon:name}));
    assert.equal(attributes['aria-pressed'], String(enabled));
    assert.equal(button.children[1].text, enabled ? 'Hide pose & boxes' : 'Show pose & boxes');
    assert.equal(button.children[0].icon, enabled ? 'eye' : 'eyeOff');
    assert.match(button.title, /Detection continues/);
  }
});
