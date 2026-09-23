'use strict';
let D, heatMeta, renderVersion=0, caseVersion=0, mapReady=false;
const $=id=>document.getElementById(id), heatCache=new Map();
const modelNames={resnet18:'ResNet-18',dinov2:'DINOv2',clip:'CLIP'};
const metricNames={l1:'L1 (raw features)',l2:'L2 (raw features)',cosine:'Cosine (unit directions)'};
const metricNotes={l1:'L1 sums absolute differences between unnormalized embeddings; feature magnitude affects this ranking.',l2:'L2 is Euclidean distance between unnormalized embeddings; both length and direction affect this ranking. It need not agree with cosine.',cosine:'Cosine distance is 1 minus the dot product of unit-normalized embeddings. It ignores vector length. Normalizing both query and reference vectors to unit length guarantees the same ranking as L2; raw L2 can also agree for particular data, but is not generally equivalent.'};
function imageCard(row,path,caption){
  const figure=document.createElement('figure');figure.className='image-card';figure.dataset.id=row.id;
  const img=document.createElement('img');img.src=path;img.alt=row.id;
  const cap=document.createElement('figcaption');cap.textContent=caption;
  const small=document.createElement('small');small.textContent=row.id;cap.append(small);figure.append(img,cap);return figure;
}
function selection(){return {qi:Number($('query').value),model:$('model').value,metric:$('metric').value,condition:$('condition').value,k:Number($('k').value)};}
function render(){
  const s=selection(),q=D.queries[s.qi],key=`${s.model}|${s.metric}|${s.condition}`,rank=D.ranks[key][s.qi],ds=D.distances[key][s.qi];
  const dogs=rank.slice(0,s.k).filter(i=>D.reference[i].source_label==='dog').length;
  $('prediction').textContent=`KNN predicts ${dogs>s.k/2?'dog':'cat'} · source label ${q.source_label}`;
  $('vote').textContent=`${dogs} dog votes, ${s.k-dogs} cat votes. Condition: ${s.condition}.`;
  $('selection-status').textContent=`Updated · ${q.id} · ${modelNames[s.model]} · ${metricNames[s.metric]} · ${s.condition} · k=${s.k}`;
  $('distance-note').textContent=metricNotes[s.metric];
  $('lab').dataset.configuration=key+'|'+s.qi+'|'+s.k;
  $('query-card').replaceChildren(imageCard(q,s.condition==='clean'?`images/${q.id}.png`:`${s.condition}/${q.id}.png`,'Locked query'));
  for(const [id,list] of [['nearest',rank.slice(0,5)],['farthest',rank.slice(-5).reverse()]]){
    $(id).replaceChildren(...list.map((i,n)=>{const card=imageCard(D.reference[i],`images/${D.reference[i].id}.png`,`${D.reference[i].source_label} · d=${ds[i].toFixed(4)}`);
      const button=document.createElement('button');button.textContent='Explain this pair';button.disabled=s.model!=='resnet18';button.addEventListener('click',()=>{$('heatmap-pair').value=String(id==='nearest'?n:n+5);$('heatmap-toggle').checked=true;renderHeatmaps();});card.append(button);return card;}));
  }
  $('voters').replaceChildren(...rank.slice(0,s.k).map((i,n)=>{const li=document.createElement('li');li.textContent=`#${n+1} ${D.reference[i].id} (${D.reference[i].source_label})`;return li;}));
  $('heatmap-toggle').disabled=s.model!=='resnet18';$('heatmap-pair').disabled=s.model!=='resnet18';
  $('heatmap-note').textContent=s.model==='resnet18'?'Grad-CAM explains the selected ResNet pair: cosine similarity, or negative raw-feature L1/L2 distance. Warm colours show positive contributions. Each heatmap is scaled independently for display; this is separate from embedding normalization and does not explain the discrete KNN vote.':'Grad-CAM is implemented for ResNet-18 only. Select ResNet-18 to inspect its own pairwise evidence; no heatmap is attributed to this encoder.';
  renderHeatmaps();updateMapSelection();
}
async function heatBytes(condition,metric){
  const key=condition+'-'+metric;
  if(!heatCache.has(key))heatCache.set(key,fetch(`heatmaps/${key}.bin`).then(r=>{if(!r.ok)throw Error(`Heatmap data: HTTP ${r.status}`);return r.arrayBuffer();}).then(b=>new Uint8Array(b)).catch(e=>{heatCache.delete(key);throw e;}));
  return heatCache.get(key);
}
async function overlay(path,bytes,offset){
  const img=new Image();img.src=path;await img.decode();
  const canvas=document.createElement('canvas');canvas.width=224;canvas.height=224;canvas.setAttribute('role','img');canvas.setAttribute('aria-label','Pairwise Grad-CAM overlay');
  const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0,224,224);const pixels=ctx.getImageData(0,0,224,224);
  for(let y=0;y<224;y++)for(let x=0;x<224;x++){
    const sx=Math.max(0,Math.min(6,(x+.5)*7/224-.5)),sy=Math.max(0,Math.min(6,(y+.5)*7/224-.5)),x0=Math.floor(sx),y0=Math.floor(sy),x1=Math.min(6,x0+1),y1=Math.min(6,y0+1),dx=sx-x0,dy=sy-y0;
    const val=Math.round((1-dy)*((1-dx)*bytes[offset+y0*7+x0]+dx*bytes[offset+y0*7+x1])+dy*((1-dx)*bytes[offset+y1*7+x0]+dx*bytes[offset+y1*7+x1]));
    const color=heatMeta.palette[val],i=(y*224+x)*4;for(let c=0;c<3;c++)pixels.data[i+c]=Math.round(.55*pixels.data[i+c]+.45*color[c]);
  }
  ctx.putImageData(pixels,0,0);return canvas;
}
async function pairPanel(qi,condition,metric,pair){
  const key=`resnet18|${metric}|${condition}`,rank=D.ranks[key][qi],ri=pair<5?rank[pair]:rank[rank.length-1-(pair-5)],q=D.queries[qi],r=D.reference[ri];
  const bytes=await heatBytes(condition,metric),offset=(qi*10+pair)*2*49;
  const paths=[`heatmaps/views/${condition}/${q.id}.jpg`,`heatmaps/views/reference/${r.id}.jpg`];
  const group=document.createElement('div');group.className='heatmap-grid';group.dataset.query=q.id;group.dataset.reference=r.id;
  const canvases=await Promise.all(paths.map((path,i)=>overlay(path,bytes,offset+i*49)));
  for(const [i,row] of [q,r].entries()){
    group.append(imageCard(row,paths[i],i===0?'Query · actual encoder crop':'Reference · actual encoder crop'));
    const f=document.createElement('figure');f.className='image-card';const cap=document.createElement('figcaption');cap.textContent=(i===0?'Query':'Reference')+' · positive pair evidence';f.append(canvases[i],cap);group.append(f);
  }
  return {group,label:`${q.id} ↔ ${r.id} · ${pair<5?'nearest':'farthest'} #${pair%5+1} · ${metricNames[metric]} distance ${D.distances[key][qi][ri].toFixed(4)}`};
}
async function renderHeatmaps(){
  const version=++renderVersion,s=selection();$('heatmap-output').replaceChildren();
  if(s.model!=='resnet18'||!$('heatmap-toggle').checked){$('heatmap-status').textContent='';return;}
  $('heatmap-status').textContent='Loading pairwise heatmap…';
  try{const result=await pairPanel(s.qi,s.condition,s.metric,Number($('heatmap-pair').value));if(version!==renderVersion)return;$('heatmap-output').replaceChildren(result.group);$('heatmap-status').textContent=result.label;}
  catch(e){if(version===renderVersion)$('heatmap-status').textContent='Unable to load heatmap: '+e.message;}
}
async function renderCase(){
  const version=++caseVersion,qi=Number($('case-query').value),rank=D.ranks['resnet18|cosine|clean'][qi],q=D.queries[qi],dogs=rank.slice(0,5).filter(i=>D.reference[i].source_label==='dog').length;
  $('case-status').textContent=`${q.id} · source ${q.source_label} · predicts ${dogs>2?'dog':'cat'} · ${dogs} dog / ${5-dogs} cat votes · clean ResNet-18 / cosine / k=5`;
  $('case-voters').replaceChildren(...rank.slice(0,5).map(i=>imageCard(D.reference[i],`images/${D.reference[i].id}.png`,D.reference[i].source_label)));
  $('case-heatmaps').replaceChildren();
  try{const result=await pairPanel(qi,'clean','cosine',Number($('case-pair').value));if(version!==caseVersion)return;$('case-heatmaps').replaceChildren(result.group);$('case-pair-status').textContent=result.label;}
  catch(e){if(version===caseVersion)$('case-pair-status').textContent=e.message;}
}
async function maps(){
  for(const model of ['resnet18','dinov2','clip']){
    const xy=D.maps[model],all=[...D.reference,...D.queries],div=$(`map-${model}`),traces=[];
    for(const label of ['cat','dog'])for(const split of ['reference','test']){
      const ids=all.map((r,i)=>r.source_label===label&&r.split===split?i:-1).filter(i=>i>=0);
      traces.push({type:'scatter',mode:'markers',name:`${label} · ${split==='test'?'query':'reference'}`,x:ids.map(i=>xy[i][0]),y:ids.map(i=>xy[i][1]),customdata:ids.map(i=>i-200),text:ids.map(i=>all[i].id+' · '+label),hovertemplate:'%{text}<extra></extra>',marker:{size:split==='test'?6:5,color:label==='cat'?'#266a9f':'#ba6031',symbol:split==='test'?'star':'circle',opacity:.65}});
    }
    traces.push({type:'scatter',mode:'markers',name:'Selected query',x:[],y:[],hoverinfo:'skip',marker:{size:17,symbol:'circle-open',color:'#162b24',line:{width:3}}});
    await Plotly.newPlot(div,traces,{title:{text:modelNames[model]},height:420,margin:{l:40,r:15,t:45,b:105},legend:{orientation:'h',y:-.22,font:{size:10}},paper_bgcolor:'#f5f4ef',plot_bgcolor:'#f5f4ef',xaxis:{title:{text:'UMAP 1 (unitless)'}},yaxis:{title:{text:'UMAP 2 (unitless)'}}},{responsive:true,displayModeBar:false});
    div.on('plotly_click',e=>{const i=e.points[0].customdata;if(Number.isInteger(i)&&i>=0){$('query').value=i;render();$('map-selection').textContent=`Selected ${D.queries[i].id}. Its clean location is ringed in all panels; retrieval above has updated.`;}else $('map-selection').textContent='This is a reference point. Select a star-shaped test query to update the retrieval controls.';});
  }
  mapReady=true;updateMapSelection();
}
function updateMapSelection(){if(!mapReady)return;const qi=Number($('query').value),i=200+qi;$('map-selection').textContent=`Selected ${D.queries[qi].id}. Its clean location is ringed in all panels; retrieval above has updated.`;for(const model of ['resnet18','dinov2','clip']){const xy=D.maps[model][i];Plotly.restyle($(`map-${model}`),{x:[[xy[0]]],y:[[xy[1]]]},[4]);}}
function row(values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;}
function option(select,value,text){const op=document.createElement('option');op.value=value;op.textContent=text;select.append(op);}
async function gunzipJson(response){
  const encoding=(response.headers.get('content-encoding')||'').toLowerCase();
  if(encoding.includes('gzip'))return response.json();
  if(typeof DecompressionStream!=='function')throw Error('This browser cannot decompress assets/data.json.gz; use a current Chrome, Edge, Firefox or Safari.');
  const stream=response.body.pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(stream).text());
}
async function start(){
  try{
    const response=await fetch('assets/data.json.gz');if(!response.ok)throw Error(response.status);D=await gunzipJson(response);
    const heatResponse=await fetch('heatmaps/index.json');if(!heatResponse.ok)throw Error('Missing heatmap export');heatMeta=await heatResponse.json();
    for(const op of $('metric').options){const value=op.value;op.value=value;op.textContent=metricNames[value];}
    const guide=document.createElement('div');guide.id='normalization-guide';
    guide.innerHTML='<p id="distance-note" role="status"></p><details><summary>Why can L2 and cosine retrieve different neighbours?</summary><p>The explorer uses raw embeddings for L1 and L2. Cosine normalizes both embeddings internally. For nonzero vectors u = x / ||x||₂ and v = y / ||y||₂, <strong>||u − v||₂² = 2(1 − u · v) = 2 × cosine distance</strong>. Unit-L2 and cosine therefore have the same ordering, subject to numerical ties. Raw L2 also depends on feature lengths, which vary across these images.</p><p>Pixel mean/standard-deviation preprocessing and model LayerNorm do not make the final embedding unit length. This includes the projected CLIP features used here. In the metric records, <strong>raw</strong> means no extra embedding normalization before the distance function; cosine still normalizes internally. <strong>unit</strong> means both embeddings are explicitly L2-normalized first. The clean unit-feature ablation is reported in PDF section B2.</p><p id="normalization-verification"></p><a href="assets/normalization_check.json">Download the numerical verification</a></details>';
    $('selection-status').after(guide);
    const geometryResponse=await fetch('assets/normalization_check.json');if(!geometryResponse.ok)throw Error('Missing normalization verification');const geometry=await geometryResponse.json();
    if(geometry.protocol_hash!==D.protocol_hash||geometry.status!=='passed')throw Error('Normalization verification does not match the protocol');
    const clean=geometry.configurations.filter(r=>r.condition==='clean');
    $('normalization-verification').textContent=`Float64 verification: ${geometry.configurations.length} encoder/condition configurations, each with 500 queries and 200 seed-1001 references. After normalizing both sides, full L2/cosine rank mismatches: ${geometry.configurations.reduce((sum,r)=>sum+r.unit_full_order_mismatches,0)}. For raw L2 versus cosine, the ordered top five differ on ${clean.map(r=>`${modelNames[r.model]} ${r.raw_top5_order_mismatches}/500`).join('; ')} clean queries. A changed order need not change the neighbour set or predicted class. This is a numerical diagnostic, not a new performance comparison.`;
    D.queries.forEach((r,i)=>option($('query'),i,r.id+' · '+r.source_label));
    for(const c of ['clean',...Object.keys(D.protocol.conditions).filter(c=>c!=='clean')])option($('condition'),c,c);
    for(const id of ['query','model','metric','condition','k'])$(id).addEventListener('change',render);
    for(let pair=0;pair<10;pair++)option($('heatmap-pair'),pair,`${pair<5?'Nearest':'Farthest'} #${pair%5+1}`);
    for(let pair=0;pair<5;pair++)option($('case-pair'),pair,`Neighbour #${pair+1}`);
    $('heatmap-toggle').addEventListener('change',renderHeatmaps);$('heatmap-pair').addEventListener('change',renderHeatmaps);
    const display=new Set(D.protocol.display_ids);let errors=0;
    D.queries.forEach((q,i)=>{const rank=D.ranks['resnet18|cosine|clean'][i],dogs=rank.slice(0,5).filter(j=>D.reference[j].source_label==='dog').length,isError=(dogs>2?'dog':'cat')!==q.source_label;
      if(display.has(q.id))option($('case-query'),i,'Preselected · '+q.id);
      else if(isError&&errors++<4)option($('case-query'),i,'Displayed error · '+q.id);
    });
    $('case-query').addEventListener('change',renderCase);$('case-pair').addEventListener('change',renderCase);
    $('inspect-case').addEventListener('click',()=>{$('query').value=$('case-query').value;$('model').value='resnet18';$('metric').value='cosine';$('condition').value='clean';$('k').value='5';$('heatmap-toggle').checked=true;render();$('lab').scrollIntoView({behavior:'smooth'});});
    $('comparisons').append(...D.evaluation.comparisons.map(r=>row([r.a+' − '+r.b,(r.difference*100).toFixed(2),r.ci.map(x=>(x*100).toFixed(2)).join(' to '),r.holm_p.toFixed(4)])));
    $('metrics').append(...D.evaluation.metrics.map(r=>row([r.key,r.accuracy.toFixed(4),r.accuracy_ci.map(x=>x.toFixed(4)).join(' to '),r.seed_sd.toFixed(4),r.macro_f1.toFixed(4),r.p5.toFixed(4)])));
    const counts={};D.audit.rows.forEach(r=>counts[r.flag]=(counts[r.flag]||0)+1);$('audit-summary').textContent='Independent audit flags: '+Object.entries(counts).map(([k,v])=>`${k}: ${v}`).join('; ')+'. Labels changed: 0.';
    $('protocol-hash').textContent=D.protocol_hash;$('test-hash').textContent=D.protocol.test_manifest_sha256;
    render();await Promise.all([maps(),renderCase()]);$('status').textContent='Locked data loaded · 500 queries · 200 display references · pairwise heatmaps ready';
  }catch(e){$('status').textContent='Load failed: '+e.message+'. Open this site over HTTP (not file://), then refresh.';console.error(e);}
}
start();
