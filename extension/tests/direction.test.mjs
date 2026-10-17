import test from 'node:test';
import assert from 'node:assert/strict';
import {validateDirection, levelsAt, channelAngle, borderPoint} from '../direction.mjs';
import {handlePlayerRequest} from '../worker.mjs';

const track = {duration:2,mode:'stereo',channels:['FL','FR'],frames:[
  {time:0,levels:{FL:1,FR:0}}, {time:1,levels:{FL:0,FR:1}}
]};

test('direction consumer validates unavailable, stereo and surround tracks',()=>{
  assert.equal(validateDirection(track),track);
  validateDirection({duration:2,mode:'unavailable',channels:[],frames:[]});
  validateDirection({duration:2,mode:'surround51',channels:['FL','FR','FC','SL','SR'],frames:[{time:0,levels:{FL:0,FR:0,FC:1,SL:0,SR:1}}]});
  for(const change of [{duration:NaN},{mode:'360'},{channels:['FL','LFE']},{frames:[]},
    {frames:[{time:0,levels:{FL:1,FR:Infinity}}]}, {frames:[{time:0,levels:{FL:1,FR:0,LFE:1}}]},
    {frames:[track.frames[0],track.frames[0]]}])assert.throws(()=>validateDirection({...track,...change}));
});

test('direction follows media time with interpolation and random-access seeking',()=>{
  assert.deepEqual(levelsAt(track,.5),{FL:.5,FR:.5});
  assert.deepEqual(levelsAt(track,1.5),{FL:0,FR:1});
  assert.deepEqual(levelsAt(track,0),{FL:1,FR:0});
  for(const time of [-1,2,NaN])assert.deepEqual(levelsAt(track,time),{});
  assert.deepEqual(levelsAt({mode:'unavailable'},1),{});
});

test('surround positions preserve nominal speaker angles across box aspect ratios',()=>{
  assert.equal(channelAngle('surround51','FL'),-30);
  assert.equal(channelAngle('surround51','SL'),-110);
  assert.equal(channelAngle('surround51','FC'),0);
  assert.equal(channelAngle('surround51','LFE'),undefined);
  for(const [width,height] of [[200,40],[80,200],[480,80]]){
    for(const channel of ['FL','FR','FC','SL','SR']){
      const angle=channelAngle('surround51',channel), p=borderPoint(width,height,angle);
      const actual=Math.atan2(p.x-width/2,-(p.y-height/2))*180/Math.PI;
      assert.ok(Math.abs(actual-angle)<1e-8);
      assert.ok(p.x>=-1e-8 && p.x<=width+1e-8 && p.y>=-1e-8 && p.y<=height+1e-8);
    }
    assert.ok(Math.abs(borderPoint(width,height,-90).x)<1e-8);
    assert.ok(Math.abs(borderPoint(width,height,90).x-width)<1e-8);
  }
});

test('worker allows direction only through the verified video resource',async()=>{
  const id='aB3dE_fG-12',url=`https://www.youtube.com/watch?v=${id}`;
  const sender={id:'extension',frameId:0,tab:{id:1},url};
  const deps={extensionId:'extension',getTab:async()=>({url}),preferences:async()=>({enabled:true,server:'http://localhost:8000'}),
    fetcher:async address=>{assert.equal(address,`http://localhost:8000/api/videos/youtube_${id}/direction`);return Response.json(track)}};
  assert.deepEqual(await handlePlayerRequest({action:'direction',youtubeId:id},sender,deps),{direction:track});
  await assert.rejects(handlePlayerRequest({action:'direction',youtubeId:id},{...sender,frameId:1},deps));
  await assert.rejects(handlePlayerRequest({action:'direction',youtubeId:id},sender,{...deps,fetcher:async()=>Response.json({...track,channels:['LFE']})}));
});
