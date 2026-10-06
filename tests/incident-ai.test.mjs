import test from 'node:test';
import assert from 'node:assert/strict';
import {IncidentAIUI, canAnalyze} from '../vmd/static/incident-ai.mjs';
const event = (id='one') => ({id, mode:'live',review:'unreviewed',clip:'evidence.avi',camera_name:'Camera',event_type:'fight',created:100,signals:{live_camera:true}});
const report = {summary:'Visible actions',assessment:'unclear',subjects:[],timeline:[],uncertainties:[],briefing:'Please verify'};
const data = id => ({incident_id:id,status:'ready',report,approved:false,following:false,observations:[]});
async function withUI(run) {
  const previous=globalThis.document;
  const node=tag=>({tag,children:[],listeners:{},open:false,textContent:'',
    append(...children){this.children.push(...children)}, after(){}, setAttribute(){},
    replaceChildren(...children){this.children=children}, addEventListener(key,fn){this.listeners[key]=fn},
    showModal(){this.open=true},close(){this.open=false;this.listeners.close?.()}});
  globalThis.document={createElement:node,getElementById:()=>node('button'),body:node('body')};
  const ui=new IncidentAIUI({api:{request:async path=>path==='/ai/status'?{configured:true}:data(path.split('/')[2])}});
  try {await run(ui)} finally {ui.dialog?.close();clearTimeout(ui.timer);globalThis.document=previous}
}
test('cloud analysis excludes presentation, synthetic, unknown and dismissed events',()=>{
  assert.ok(canAnalyze(event()));assert.ok(canAnalyze({...event(),signals:{live_camera:false}}));
  for (const patch of [{mode:'demo'},{mode:'presentation'},{signals:{}},{clip:null},{review:'false_positive'}]) assert.equal(canAnalyze({...event(),...patch}),false);
});
test('opening reads status only; approve explicitly targets the opened incident',async()=>withUI(async ui=>{
  const requests=[];
  ui.workspace.api.request=async(path,body)=>{requests.push({path,body});return path==='/ai/status'?{configured:true}:data('one')};
  ui.open(event());await ui.refresh();
  assert.ok(requests.every(row=>row.body===undefined));
  await ui.action('/approve',{enabled:true});
  assert.deepEqual(requests.at(-1),{path:'/incidents/one/ai/approve',body:{enabled:true}});
  assert.equal(ui.download.href,'/api/incidents/one/ai/report.pdf');
  assert.equal(ui.jsonDownload.href,'/api/incidents/one/ai/report');
}));
test('late action response cannot overwrite another incident',async()=>withUI(async ui=>{
  ui.open(event());await ui.refresh();let resolve;
  ui.workspace.api.request=()=>new Promise(done=>resolve=done);
  const pending=ui.action('',{quality:true});
  ui.session={id:'two',event:event('two'),busy:false,data:data('two'),config:{configured:true}};
  resolve(data('one'));await pending;
  assert.equal(ui.session.data.incident_id,'two');
}));
test('missing cloud credentials block uploads and closing clears report',async()=>withUI(async ui=>{
  ui.workspace.api.request=async path=>path==='/ai/status'?{configured:false,error:'Sign in locally'}:data('one');
  ui.open(event());await ui.refresh();
  assert.equal(ui.analyze.disabled,true);assert.equal(ui.askButton.disabled,true);
  assert.match(ui.status.textContent,/Sign in/);
  ui.dialog.close();assert.equal(ui.session,null);assert.equal(ui.output.children.length,0);
}));
