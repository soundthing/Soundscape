import test from 'node:test';
import assert from 'node:assert/strict';
import { parseVtt, activeCaption } from '../vtt.mjs';
import { watchVideoId } from '../settings.mjs';
import { readPreferences, updatePreferences } from '../preferences.mjs';
import { handlePlayerRequest } from '../worker.mjs';

const id='aB3dE_fG-12', url=`https://www.youtube.com/watch?v=${id}`;
const vtt='WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n[footsteps]\n\nsecond\n00:00:02.000 --> 00:00:04.000\n[door slams]\nsecond line\n';

test('VTT handles overlap, seek boundaries, multiline text, BOM, CRLF, identifiers, and hours',()=>{
  const cues=parseVtt('\uFEFF'+vtt.replaceAll('\n','\r\n'));
  assert.equal(activeCaption(cues,0),'');
  assert.equal(activeCaption(cues,1),'[footsteps]');
  assert.equal(activeCaption(cues,2),'[footsteps]\n[door slams]\nsecond line');
  assert.equal(activeCaption(cues,3),'[door slams]\nsecond line');
  assert.equal(activeCaption(cues,4),'');
  assert.equal(activeCaption(cues,1),'[footsteps]');
  assert.equal(parseVtt('WEBVTT\n\n01:00:00.000 --> 01:00:02.000\n[test]\n')[0].start,3600);
  for(const text of ['not VTT','WEBVTT\n\n00:00:03.000 --> 00:00:01.000\nwrong\n','WEBVTT\n\n00:70:00.000 --> 00:71:00.000\nwrong\n'])assert.throws(()=>parseVtt(text));
  assert.deepEqual(parseVtt('WEBVTT\n\n'),[]);
});

test('only standard www.youtube.com watch pages enable automatic processing',()=>{
  assert.equal(watchVideoId(url),id);
  for(const address of [`https://www.youtube.com/shorts/${id}`,`https://www.youtube.com/embed/${id}`,`https://youtube.com.evil/watch?v=${id}`,`http://www.youtube.com/watch?v=${id}`,'https://www.youtube.com/'])assert.equal(watchVideoId(address),null);
});

test('preferences default off and persist toggle separately from the old library',async()=>{
  const data={soundthingWorkspace:{jobs:[{name:'existing'}]}};
  const storage={async get(){return data},async set(update){Object.assign(data,update)}};
  assert.deepEqual(await readPreferences(storage),{enabled:false,server:'http://127.0.0.1:8000'});
  await updatePreferences({enabled:true},storage);
  assert.equal((await readPreferences(storage)).enabled,true);
  assert.equal(data.soundthingWorkspace.jobs[0].name,'existing');
  await assert.rejects(updatePreferences({server:'https://evil.test'},storage));
});

const sender={id:'extension',frameId:0,tab:{id:1},url};
const deps={extensionId:'extension',getTab:async()=>({url}),preferences:async()=>({enabled:true,server:'http://localhost:8000'})};

test('worker constructs local requests from verified sender IDs and returns status / VTT',async()=>{
  const calls=[];
  const options={...deps,fetcher:async(address,init)=>{
    calls.push([address,init]);
    if(address.endsWith('/api/youtube'))return Response.json({video_id:`youtube_${id}`});
    if(address.endsWith('.vtt'))return new Response(vtt);
    return Response.json({status:'downloading',demo:true});
  }};
  const message={youtubeId:id,action:'ensure'};
  assert.deepEqual(await handlePlayerRequest(message,sender,options),{videoId:`youtube_${id}`});
  assert.deepEqual(JSON.parse(calls[0][1].body),{youtube_id:id});
  assert.equal((await handlePlayerRequest({...message,action:'status'},sender,options)).status,'downloading');
  assert.equal((await handlePlayerRequest({...message,action:'captions'},sender,options)).vtt,vtt);
  assert.equal(calls[2][0],`http://localhost:8000/api/videos/youtube_${id}/captions/conditioned.vtt`);
  // Chrome may keep the original document URL after an SPA navigation.
  assert.deepEqual(await handlePlayerRequest(message,{...sender,url:'https://www.youtube.com/'},options),{videoId:`youtube_${id}`});
});

test('worker rejects foreign frames, stale videos, disabled state, and arbitrary fetch actions',async()=>{
  const fetcher=()=>{throw Error('must not fetch')};
  const message={youtubeId:id,action:'ensure'};
  for(const bad of [{...sender,id:'foreign'},{...sender,frameId:1},{...sender,url:'https://evil.test'},{...sender,tab:undefined}])await assert.rejects(handlePlayerRequest(message,bad,{...deps,fetcher}));
  await assert.rejects(handlePlayerRequest({...message,youtubeId:'xY3dE_fG-12'},sender,{...deps,fetcher}),/changed/);
  await assert.rejects(handlePlayerRequest({...message,action:'fetch',url:'https://evil.test'},sender,{...deps,fetcher}),/Unknown/);
  await assert.rejects(handlePlayerRequest(message,sender,{...deps,fetcher,preferences:async()=>({enabled:false})}),/off/);
});
