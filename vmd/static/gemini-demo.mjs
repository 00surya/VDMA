const $=id=>document.getElementById(id);
let pending=false,ready=false,asking=false;
async function request(path,body){
  const response=await fetch('/api/'+path,{...(body?{method:'POST',headers:{'Content-Type':'application/json','X-VMD-Client':'dashboard'},body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(60000)});
  const value=await response.json();if(!response.ok)throw new Error(value.detail||'Request failed');return value;
}
function text(tag,value){const node=document.createElement(tag);node.textContent=value;return node;}
function paint(data){
  const report=data.report;$('report').replaceChildren();
  if(!report){$('report').append(text('p',data.error||'Waiting for actual Gemini observations…'));return;}
  $('report').append(text('p',report.summary),text('p','Assessment: '+report.assessment.replaceAll('_',' ')));
  $('report').append(text('h3','Visible subjects'));const subjects=document.createElement('ul');report.subjects.forEach(value=>subjects.append(text('li',value)));$('report').append(subjects);
  $('report').append(text('h3','Timeline'));const timeline=document.createElement('ol');report.timeline.forEach(row=>timeline.append(text('li',`${row.seconds.toFixed(1)}s · ${row.observation}`)));$('report').append(timeline);
  if(report.uncertainties.length)$('report').append(text('p','Unclear: '+report.uncertainties.join(' · ')));
  $('report').append(text('h3','Draft responder briefing'),text('p',report.briefing));
  if(data.usage)$('report').append(text('p',`Model: ${data.usage.model} · ${data.usage.frames} frames · ${data.usage.input_tokens??'?'} input tokens`));
}
async function refresh(){
  try{const[setup,data]=await Promise.all([request('status'),request('report')]);ready=setup.configured;pending=['queued','running'].includes(data.status);$('status').textContent=setup.error||data.error||`Real Gemini analysis: ${data.status.replaceAll('_',' ')}`;paint(data);$('analyze').disabled=!ready||pending||asking;$('ask').disabled=!ready||pending||asking;}
  catch(error){$('status').textContent=error.message;}
}
$('analyze').addEventListener('click',async()=>{if(pending||asking)return;pending=true;$('analyze').disabled=true;try{await request('analyze',{quality:$('model').value==='quality'});await refresh();}catch(error){$('status').textContent=error.message;pending=false;$('analyze').disabled=!ready;}});
$('questions').addEventListener('submit',async event=>{event.preventDefault();if(asking||pending)return;asking=true;$('ask').disabled=true;$('analyze').disabled=true;$('answer').textContent='Gemini is examining the evidence…';try{const result=await request('ask',{question:$('question').value});$('answer').textContent=result.answer+'\n'+result.limitations;}catch(error){$('answer').textContent=error.message;}finally{asking=false;$('ask').disabled=!ready||pending;$('analyze').disabled=!ready||pending;}});
await refresh();setInterval(refresh,2500);
