import test from 'node:test';
import assert from 'node:assert/strict';
import { parseVtt, activeCaption } from '../vtt.mjs';
import { watchVideoId } from '../settings.mjs';
import { readPreferences, updatePreferences } from '../preferences.mjs';
import { handlePlayerRequest } from '../worker.mjs';
import { CaptionController, displayedCaptions } from '../player.mjs';

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

test('display adds a 0.75-second fade per cue without changing VTT timing or fading newer captions',()=>{
  const cues = Object.freeze(parseVtt(vtt).map(cue => Object.freeze(cue)));
  const original = JSON.stringify(cues);
  const at = time => displayedCaptions(cues, time).map(({cue, opacity}) => [cue.text, opacity]);
  assert.deepEqual(at(.99), []);
  assert.deepEqual(at(3), [['[footsteps]', 1], ['[door slams]\nsecond line', 1]]);
  assert.deepEqual(at(3.375), [['[footsteps]', .5], ['[door slams]\nsecond line', 1]]);
  assert.deepEqual(at(3.75), [['[door slams]\nsecond line', 1]]);
  assert.deepEqual(at(4.375), [['[door slams]\nsecond line', .5]]);
  assert.deepEqual(at(4.75), []);
  // Seeking backwards restores the original line and its full opacity.
  assert.deepEqual(at(1), [['[footsteps]', 1]]);
  assert.equal(JSON.stringify(cues), original);
  assert.equal(activeCaption(cues, 4.375), ''); // Original VTT semantics remain unchanged.
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

test('console cache deletion is scoped to the current video and works while captions are off',async()=>{
  const calls=[];
  const options={...deps,preferences:async()=>({enabled:false,server:'http://localhost:8000'}),fetcher:async(address,init)=>{
    calls.push([address,init]);
    return Response.json({video_id:`youtube_${id}`,cleared:true,removed:['results','downloads']});
  }};
  const message={youtubeId:id,action:'clearCache',url};
  const result=await handlePlayerRequest(message,sender,options);
  assert.equal(result.cleared,true);
  assert.equal(calls[0][0],`http://localhost:8000/api/youtube/cache?url=${id}`);
  assert.equal(calls[0][1].method,'DELETE');
  await assert.rejects(handlePlayerRequest({...message,url:'https://youtu.be/xY3dE_fG-12'},sender,options),/Open that video's/);
  await assert.rejects(handlePlayerRequest(message,{...sender,frameId:1},options),/Unsupported/);
  await assert.rejects(handlePlayerRequest(message,sender,{...options,fetcher:async()=>Response.json({detail:'Still processing'},{status:409})}),/Still processing/);
  assert.equal(calls.length,1);
});

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

test('worker retrieves scenes through the existing verified environment endpoint',async()=>{
  const scenes=[{start:0,end:4,env:'general',confidence:.8,probs:{general:1}}];
  const options={...deps,fetcher:async address=>{
    assert.equal(address,`http://localhost:8000/api/videos/youtube_${id}/env_segments`);
    return Response.json(scenes);
  }};
  assert.deepEqual(await handlePlayerRequest({youtubeId:id,action:'scenes'},sender,options),{scenes});
  await assert.rejects(handlePlayerRequest({youtubeId:id,action:'scenes'},{...sender,frameId:1},options));
  await assert.rejects(handlePlayerRequest({youtubeId:id,action:'scenes'},sender,{...options,fetcher:async()=>Response.json({unexpected:true})}),/invalid environment scenes/);
});

test('completion logs full VTT and scenes once without affecting the ready state',async()=>{
  const messages=[],scenes=[{start:0,end:4,env:'general',confidence:1,probs:{general:1}}];
  const session={id,server:'http://localhost:8000',demo:false,status:'done'};
  let requests=0;
  const controller={window:{console:{log:(...args)=>messages.push(args)}},isCurrent:s=>s===session,
    request:async(s,action)=>{assert.equal(action,'scenes');requests++;return {scenes}}};
  await Promise.all([CaptionController.prototype.logCompletion.call(controller,session,vtt),CaptionController.prototype.logCompletion.call(controller,session,vtt)]);
  assert.equal(requests,1);
  assert.equal(messages.length,3);
  const metadata=JSON.parse(messages[0][0].split('\n').slice(1).join('\n'));
  assert.equal(metadata.youtubeId,id);
  assert.equal(metadata.demo,false);
  assert.ok(messages[1][0].endsWith(vtt));
  assert.deepEqual(JSON.parse(messages[2][0].split('\n').slice(1).join('\n')),scenes);
  assert.equal(session.status,'done');
});

test('scene diagnostics do not break captions and stale completion responses are ignored',async()=>{
  const messages=[],session={id,server:'http://localhost:8000',status:'done'};
  const controller={window:{console:{log:(...args)=>messages.push(args)}},isCurrent:()=>true,
    request:async()=>{throw Error('Scenes endpoint unavailable')}};
  await CaptionController.prototype.logCompletion.call(controller,session,vtt);
  assert.ok(messages[1][0].includes(vtt));
  assert.equal(messages[2][0],'[Soundthing] Scenes unavailable: Scenes endpoint unavailable');
  assert.equal(session.status,'done');
  let current=true,resolve;
  controller.isCurrent=()=>current;
  controller.request=()=>new Promise(r=>resolve=r);
  const pending=CaptionController.prototype.logCompletion.call(controller,{...session,completionLogged:false},vtt);
  current=false;
  resolve({scenes:[]});
  await pending;
  assert.equal(messages.length,3);
});
