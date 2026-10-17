"""Optional Chromium integration check. Uses no live YouTube account or downloads.

Run with the server/dev extras and Playwright installed:
  python extension/tests/browser_smoke.py
The real extension, worker, HTTP API and VTT writer run against controlled
YouTube DOM/media clocks and a fixture downloader/model processor.
"""
import importlib
import tempfile
import threading
import time
from pathlib import Path

import uvicorn
from playwright.sync_api import sync_playwright, expect
from fastapi.responses import JSONResponse
from soundthing import contracts as C
from soundthing.captions import write_captions
from soundthing.server.youtube import YouTubeJobs

ROOT = Path(__file__).resolve().parents[2]
ID, OTHER = "aB3dE_fG-12", "xY3dE_fG-12"
HTML = """<!doctype html><html><head><title>Owned video fixture</title></head><body style='margin:40px;background:#181818;color:white'>
<div id='movie_player' style='position:relative;width:960px;height:540px;background:#302b37'>
<video src="MEDIA_SOURCE" preload="auto"></video><div id='speech'>YouTube speech captions stay intact</div></div>
</body></html>"""


def main():
    work = Path(tempfile.mkdtemp(prefix="soundthing-overlay-smoke-"))
    backend = importlib.import_module("soundthing.server.app")
    backend.RUNS, backend.UPLOADS = work / "runs", work / "uploads"
    backend.status = {}
    downloads = []
    outage = {"offline": False, "missing_status": None, "direction_failures": 2}

    @backend.app.middleware("http")
    async def simulate_restart(request, call_next):
        if request.url.path.endswith("/direction") and outage["direction_failures"]:
            outage["direction_failures"] -= 1
            return JSONResponse({"detail": "direction not ready"}, status_code=503 if outage["direction_failures"] else 404)
        if outage["offline"] and request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "temporary test outage"}, status_code=503)
        if request.url.path == outage["missing_status"]:
            outage["missing_status"] = None
            return JSONResponse({"detail": "job lost during test restart"}, status_code=404)
        return await call_next(request)

    def download(video_id, uploads):
        downloads.append(video_id)
        time.sleep(.25)
        uploads.mkdir(parents=True, exist_ok=True)
        source = uploads / f"youtube_{video_id}.mp4"
        source.write_bytes(b"fixture")
        return source

    def process(source, runs, cfg):
        time.sleep(.25)
        run = runs / source.stem
        run.mkdir(parents=True, exist_ok=True)
        C.save_inputs(C.ExtractedInputs(video_id=source.stem, video_path=str(source), audio_path=str(source), frames_dir=str(run), fps=1, duration=10), run / "inputs.json")
        cls = cfg.classes[0]
        events = [C.CaptionEvent(start=1, end=3, label=cls.caption, class_id=cls.id, confidence=.9)]
        for mode in ("baseline", "conditioned"):
            write_captions(events, run / f"captions.{mode}.vtt")
        if source.stem == f"youtube_{OTHER}":
            channels = ["FL", "FR", "FC", "SL", "SR"]
            track = C.DirectionTrack(10, "surround51", channels, [C.DirectionFrame(0, {"FL": 0, "FR": 0, "FC": .9, "SL": 0, "SR": .8})])
        else:
            track = C.DirectionTrack(10, "stereo", ["FL", "FR"], [
                C.DirectionFrame(0, {"FL": .9, "FR": 0}), C.DirectionFrame(1.9, {"FL": .9, "FR": 0}),
                C.DirectionFrame(2, {"FL": 0, "FR": .9})])
        C.save_direction(track, run / "direction.json")

    backend.youtube_jobs = YouTubeJobs(backend.RUNS, backend.UPLOADS, backend.cfg, backend.status, downloader=download, processor=process)
    server = uvicorn.Server(uvicorn.Config(backend.app, host="127.0.0.1", port=8766, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(.05)
    assert server.started
    try:
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(str(work / "browser"), channel="chromium", headless=True, viewport={"width":1200,"height":800}, args=[f"--disable-extensions-except={ROOT / 'extension'}", f"--load-extension={ROOT / 'extension'}"])
            context.set_default_timeout(10000)
            maker = context.new_page()
            media = maker.evaluate("""async () => {
              const canvas=document.createElement('canvas');canvas.width=320;canvas.height=180;
              const ctx=canvas.getContext('2d');ctx.fillRect(0,0,320,180);
              const stream=canvas.captureStream(10),recorder=new MediaRecorder(stream,{mimeType:'video/webm'}),chunks=[];
              recorder.ondataavailable=e=>chunks.push(e.data);
              const stopped=new Promise(resolve=>recorder.onstop=resolve);
              recorder.start(); const timer=setInterval(()=>{ctx.fillStyle=`hsl(${Date.now()%360},40%,30%)`;ctx.fillRect(0,0,320,180)},100);
              await new Promise(resolve=>setTimeout(resolve,4500));recorder.stop();await stopped;clearInterval(timer);stream.getTracks().forEach(t=>t.stop());
              const bytes=new Uint8Array(await new Blob(chunks).arrayBuffer());return btoa(String.fromCharCode(...bytes));
            }""")
            maker.close()
            fixture_html=HTML.replace('MEDIA_SOURCE','data:video/webm;base64,'+media)
            context.route("https://www.youtube.com/**", lambda route: route.fulfill(content_type="text/html", body=fixture_html))
            sw = context.service_workers[0] if context.service_workers else context.wait_for_event("serviceworker")
            origin = "/".join(sw.url.split("/")[:3])
            errors = []
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"https://www.youtube.com/watch?v={ID}")
            popup = context.new_page()
            popup.goto(f"{origin}/popup.html")
            expect(popup.locator("#enabled")).not_to_be_checked()
            assert not downloads
            popup.locator("summary").click()
            popup.locator("#server").fill("http://127.0.0.1:8766")
            popup.locator("#connection button").click()
            expect(popup.locator("#connection-message")).to_contain_text("Connected")
            popup.locator("#enabled").check()
            popup.close()  # Popup lifetime must not own downloads or polling.
            page.bring_to_front()
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false", timeout=15000)
            page.wait_for_function("document.querySelector('video').readyState >= 2")
            page.evaluate("document.querySelector('video').currentTime=1.5")
            overlay = page.locator("#soundthing-caption-overlay")
            expected = f"[{backend.cfg.classes[0].caption}]"
            expect(overlay.locator("div")).to_contain_text(expected)
            expect(page.locator("#speech")).to_have_text("YouTube speech captions stay intact")
            expect(overlay.locator("svg")).to_have_attribute("data-direction", "stereo", timeout=15000)
            expect(overlay.locator('[data-channel="FL"]')).to_be_visible()
            expect(overlay.locator('[data-channel="FR"]')).to_be_hidden()
            assert not outage["direction_failures"]
            frozen = overlay.locator('[data-channel="FL"]').get_attribute("d")
            page.wait_for_timeout(150)
            assert overlay.locator('[data-channel="FL"]').get_attribute("d") == frozen
            page.emulate_media(reduced_motion="reduce")
            page.wait_for_timeout(100)
            static = overlay.locator('[data-channel="FL"]').get_attribute("d")
            page.evaluate("document.querySelector('video').currentTime=1.7")
            page.wait_for_timeout(100)
            assert overlay.locator('[data-channel="FL"]').get_attribute("d") == static
            page.emulate_media(reduced_motion="no-preference")
            print("PASS: stereo waves, independent recovery from direction 503/404, pause and reduced motion", flush=True)
            page.screenshot(path=str(work / "overlay.png"))
            print("PASS: global toggle, real worker/API/VTT overlay, and closing the popup during processing", flush=True)
            page.evaluate("document.querySelector('video').currentTime=4")
            expect(overlay).to_be_hidden()
            page.evaluate("document.querySelector('video').currentTime=2;document.querySelector('video').playbackRate=2")
            expect(overlay.locator("div")).to_contain_text(expected)
            expect(overlay.locator('[data-channel="FR"]')).to_be_visible()
            expect(overlay.locator('[data-channel="FL"]')).to_be_hidden()
            page.evaluate("document.querySelector('#movie_player').classList.add('ad-showing')")
            expect(overlay).to_be_hidden()
            page.evaluate("document.querySelector('#movie_player').classList.remove('ad-showing')")
            expect(overlay).to_be_visible()
            page.evaluate("document.querySelector('#movie_player').style.width='1120px'")
            box = overlay.bounding_box()
            parent = page.locator("#movie_player").bounding_box()
            assert box["x"] > parent["x"] + parent["width"] / 2
            assert box["y"] > parent["y"] + parent["height"] / 2
            print("PASS: seek/speed timing, ad suppression, preserved speech captions, and bottom-right placement", flush=True)
            second = context.new_page()
            second.goto(f"https://www.youtube.com/watch?v={ID}")
            second.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false")
            assert downloads.count(ID) == 1
            page.bring_to_front()
            page.evaluate(f"history.pushState({{}},'', '/watch?v={OTHER}');document.dispatchEvent(new Event('yt-navigate-finish'))")
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false")
            assert OTHER in downloads
            page.evaluate("document.querySelector('video').currentTime=1.5")
            expect(overlay.locator("svg")).to_have_attribute("data-direction", "surround51")
            expect(overlay.locator('[data-channel="FC"]')).to_be_visible()
            expect(overlay.locator('[data-channel="SR"]')).to_be_visible()
            assert overlay.locator('[data-channel="LFE"]').count() == 0
            page.screenshot(path=str(work / "surround-waves.png"))
            print("PASS: automatic stereo-to-5.1 switch, independent front/surround waves, and no LFE direction", flush=True)
            print("PASS: cross-tab deduplication and automatic processing on YouTube SPA navigation", flush=True)
            page.evaluate("async () => { await document.querySelector('#movie_player').requestFullscreen(); }")
            page.wait_for_function("document.fullscreenElement?.id === 'movie_player'")
            assert page.evaluate("document.fullscreenElement.contains(document.querySelector('#soundthing-caption-overlay'))")
            page.evaluate("document.exitFullscreen()")
            page.reload()
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false")
            page.evaluate("const original=document.querySelector('video');original.replaceWith(original.cloneNode(true))")
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false")
            assert downloads.count(OTHER) == 1
            print("PASS: fullscreen containment, page refresh, and video-element replacement reuse cached jobs", flush=True)
            outage["offline"] = True
            recovering = "rR3dE_fG-12"
            page.evaluate(f"history.pushState({{}},'', '/watch?v={recovering}');document.dispatchEvent(new Event('yt-navigate-finish'))")
            expect(page.locator('#soundthing-caption-overlay').locator('div')).to_contain_text("reconnecting")
            outage["offline"] = False
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false", timeout=20000)
            restarted = "sS3dE_fG-12"
            outage["missing_status"] = f"/api/videos/youtube_{restarted}"
            page.evaluate(f"history.pushState({{}},'', '/watch?v={restarted}');document.dispatchEvent(new Event('yt-navigate-finish'))")
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false", timeout=20000)
            assert downloads.count(restarted) == 1
            print("PASS: automatic recovery from temporary server outages and missing jobs after restart", flush=True)
            popup = context.new_page()
            popup.goto(f"{origin}/popup.html")
            expect(popup.locator("#enabled")).to_be_checked()
            popup.locator("#enabled").uncheck()
            expect(page.locator("#soundthing-caption-overlay")).to_have_count(0)
            expect(second.locator("#soundthing-caption-overlay")).to_have_count(0)
            popup.screenshot(path=str(work / "popup.png"))
            print("PASS: toggle persists and disabling clears every tab", flush=True)

            # Test both developer policies against the same production controller.
            popup.evaluate("""async origin => {
              const {CaptionController}=await import(origin+'/player.mjs');
              window.makePolicyFixture=policy=>{
                window.policyController?.disposeSession();
                document.querySelector('#movie_player')?.remove();
                const player=document.createElement('div'); player.id='movie_player';
                player.innerHTML='<video></video>'; document.body.append(player);
                const video=player.querySelector('video');
                window.pausedState=false; window.playCalls=0; window.pauseCalls=0;
                Object.defineProperty(video,'paused',{get:()=>window.pausedState});
                video.pause=()=>{window.pausedState=true;window.pauseCalls++};
                video.play=()=>{window.pausedState=false;window.playCalls++;return Promise.resolve()};
                let finish; window.readyGate=new Promise(resolve=>finish=resolve); window.finishPolicy=finish;
                const fakeWindow={location:{href:'https://www.youtube.com/watch?v=aB3dE_fG-12'},requestAnimationFrame:requestAnimationFrame.bind(window),cancelAnimationFrame:cancelAnimationFrame.bind(window),setTimeout:setTimeout.bind(window),clearTimeout:clearTimeout.bind(window)};
                window.policyController=new CaptionController({window:fakeWindow,document,playbackPolicy:policy,send:async message=>{
                  if(message.action==='ensure')return {ok:true};
                  if(message.action==='status'){await window.readyGate;return {ok:true,status:'done'}};
                  return {ok:true,vtt:'WEBVTT\\n\\n00:00:00.000 --> 00:00:05.000\\n[test]\\n'};
                }});
                window.policyController.configure({enabled:true,server:'http://localhost:8766'});
              };
            }""", origin)
            popup.evaluate("makePolicyFixture('keep-playing')")
            assert popup.evaluate("pauseCalls") == 0
            popup.evaluate("finishPolicy()")
            popup.wait_for_function("policyController.state.status==='done'")
            popup.evaluate("makePolicyFixture('pause-until-ready')")
            assert popup.evaluate("pausedState && pauseCalls===1")
            popup.evaluate("finishPolicy()")
            popup.wait_for_function("policyController.state.status==='done'")
            assert popup.evaluate("!pausedState && playCalls===1")
            popup.evaluate("makePolicyFixture('pause-until-ready')")
            popup.locator("#movie_player").dispatch_event("pointerdown")
            # Synthetic events intentionally do not claim to be user interactions.
            popup.locator("#movie_player").click(force=True)
            popup.evaluate("finishPolicy()")
            popup.wait_for_function("policyController.state.status==='done'")
            assert popup.evaluate("pausedState && playCalls===0")
            popup.evaluate("makePolicyFixture('pause-until-ready');policyController.configure({enabled:false,server:'http://localhost:8766'})")
            assert popup.evaluate("!pausedState && playCalls===1")
            popup.evaluate("finishPolicy()")
            print("PASS: both playback policies, user-intervention protection, and resume on disable", flush=True)
            assert not errors, errors
            context.close()
            print(f"All browser smoke checks passed. Screenshots: {work}", flush=True)
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
