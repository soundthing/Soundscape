"""Optional Chromium integration check. Uses no live YouTube account or downloads.

Run with the server/dev extras and Playwright installed:
  python extension/tests/browser_smoke.py
The real extension, worker, HTTP API and VTT writer run against controlled
YouTube DOM/media clocks and a fixture downloader/model processor.
"""
import importlib
import re
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


def wave_bounds(overlay):
    geometry = overlay.locator('[data-wave="border"]').get_attribute("d")
    assert geometry.endswith(" Z")
    points = [(float(x), float(y)) for x, y in re.findall(r"[ML](-?[\d.]+),(-?[\d.]+)", geometry)]
    return min(x for x, y in points), min(y for x, y in points), max(x for x, y in points), max(y for x, y in points)


def main():
    work = Path(tempfile.mkdtemp(prefix="soundthing-overlay-smoke-"))
    backend = importlib.import_module("soundthing.server.app")
    backend.RUNS, backend.UPLOADS = work / "runs", work / "uploads"
    backend.status = {}
    downloads = []
    outage = {"offline": False, "missing_status": None, "direction_failures": 2, "offline_requests": 0}

    @backend.app.middleware("http")
    async def simulate_restart(request, call_next):
        if request.url.path.endswith("/direction") and outage["direction_failures"]:
            outage["direction_failures"] -= 1
            return JSONResponse({"detail": "direction not ready"}, status_code=503 if outage["direction_failures"] else 404)
        if outage["offline"] and request.url.path.startswith("/api/"):
            outage["offline_requests"] += 1
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
        events = [C.CaptionEvent(start=1, end=3, label=cls.caption, class_id=cls.id, confidence=.9),
                  C.CaptionEvent(start=2.2, end=2.8, label=" · ".join([cls.caption] * 8), class_id=cls.id, confidence=.9)]
        for mode in ("baseline", "conditioned"):
            C.save_events(events, run / f"events.{mode}.json")
            write_captions(events, run / f"captions.{mode}.vtt")
        environment = cfg.environments[0].id
        C.save_segments([C.EnvSegment(0, 10, environment, 1, {environment: 1})], run / "env_segments.json")
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
            debug_logs = []
            page = context.new_page()
            cdp = context.new_cdp_session(page)
            execution_contexts = {}
            cdp.on("Runtime.executionContextCreated", lambda event: execution_contexts.update({event["context"]["id"]: event["context"]}))
            cdp.on("Runtime.executionContextDestroyed", lambda event: execution_contexts.pop(event["executionContextId"], None))
            cdp.on("Runtime.executionContextsCleared", lambda event: execution_contexts.clear())
            cdp.send("Runtime.enable")
            page.on("pageerror", lambda error: errors.append(str(error)))
            def capture_debug(message):
                if message.type == "log" and message.text.startswith("[Soundthing]"):
                    debug_logs.append(message.text)
            page.on("console", capture_debug)
            page.on("console", lambda msg: print("BROWSER:", msg.text, flush=True) if msg.type in ("warning", "error") else None)
            page.goto(f"https://www.youtube.com/watch?v={ID}")
            popup = context.new_page()
            popup.goto(f"{origin}/popup.html")
            expect(popup.locator("#enabled")).not_to_be_checked()
            expect(popup.locator("h1")).to_have_text("soundscape.")
            popup.evaluate("document.fonts.ready")
            expect(popup.locator(".surf-caption")).to_be_hidden()
            wave_before = popup.locator(".shore-wave").first.evaluate("el => getComputedStyle(el).transform")
            popup.wait_for_timeout(400)
            assert popup.locator(".shore-wave").first.evaluate("el => getComputedStyle(el).transform") != wave_before
            popup.locator("body").screenshot(path=str(work / "popup-off.png"))
            popup.evaluate("() => document.querySelectorAll('.shore-wave').forEach(el=>el.getAnimations().forEach(animation=>animation.pause()))")
            assert not downloads
            # Server configuration is test setup, not part of the minimal popup.
            popup.evaluate("chrome.storage.local.set({soundthingPreferences:{enabled:false,server:'http://127.0.0.1:8766'}})")
            popup.locator("#enabled").check()
            expect(popup.locator("h1")).to_have_text("soundscape!")
            popup.wait_for_function("() => getComputedStyle(document.querySelector('#click-caption')).opacity > .8")
            positions = []
            # Extreme random values must keep the full rotated word safely inset.
            for word, value in zip(["splish", "splosh", "splish", "splash", "splish"], [0, 1, .35, .65, .2]):
                popup.evaluate("""value => {
                  const original=Math.random;
                  Math.random=()=>value;
                  try {
                    document.querySelector('.shore-wave').dispatchEvent(new AnimationEvent('animationiteration',{animationName:'shore-wash',bubbles:true}));
                  } finally { Math.random=original; }
                  const animation=document.querySelector('.surf-caption').getAnimations()[0];
                  animation.pause(); animation.currentTime=400;
                }""", value)
                expect(popup.locator(".surf-caption")).to_have_text(word)
                expect(popup.locator(".surf-caption")).to_have_css("opacity", "1")
                caption = popup.locator(".surf-caption").bounding_box()
                shore = popup.locator(".beach").bounding_box()
                assert caption["x"] >= shore["x"] + 23.5
                assert caption["y"] >= shore["y"] + 23.5
                assert caption["x"] + caption["width"] <= shore["x"] + shore["width"] - 23.5
                assert caption["y"] + caption["height"] <= shore["y"] + shore["height"] - 23.5
                positions.append(caption["x"])
            assert max(positions) - min(positions) > 100
            # Long status messages grow the controls down without moving their top.
            layout = popup.evaluate("""() => {
              const copy=document.querySelector('.control-copy'),toggle=document.querySelector('.toggle'),controls=document.querySelector('.controls');
              const before={copy:copy.getBoundingClientRect().top,toggle:toggle.getBoundingClientRect().top,height:controls.getBoundingClientRect().height};
              const detail=document.querySelector('#status-detail'),saved=detail.textContent;
              detail.textContent='Reconnecting to the local server. Start the server and retry this video once the connection has been restored.';
              const after={copy:copy.getBoundingClientRect().top,toggle:toggle.getBoundingClientRect().top,height:controls.getBoundingClientRect().height};
              detail.textContent=saved;
              return {before,after};
            }""")
            assert layout["before"]["copy"] == layout["after"]["copy"]
            assert layout["before"]["toggle"] == layout["after"]["toggle"]
            assert layout["after"]["height"] > layout["before"]["height"]
            print("PASS: ordered shore words, randomized positions with safe edges, and downward-growing status text", flush=True)
            popup.locator("body").screenshot(path=str(work / "popup-click.png"))
            popup.close()  # Popup lifetime must not own downloads or polling.
            page.bring_to_front()
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false", timeout=15000)
            page.wait_for_function("document.querySelector('video').readyState >= 2")
            page.evaluate("document.querySelector('video').currentTime=1.5")
            overlay = page.locator("#soundthing-caption-overlay")
            expected = f"[{backend.cfg.classes[0].caption}]"
            expect(overlay.locator("div")).to_contain_text(expected)
            page.wait_for_function("[...document.fonts].some(font=>font.family.includes('Soundthing Monocraft') && font.status==='loaded')")
            assert overlay.locator("header, .brand, .mode").count() == 0
            assert "Soundthing Monocraft" in overlay.locator("div").evaluate("el=>getComputedStyle(el).fontFamily")
            expect(page.locator("#speech")).to_have_text("YouTube speech captions stay intact")
            expect(overlay.locator("svg")).to_have_attribute("data-direction", "stereo", timeout=15000)
            expect(overlay.locator('[data-wave="border"]')).to_be_visible()
            assert overlay.locator("svg path").count() == 1
            left, top, right, bottom = wave_bounds(overlay)
            assert left < -2 and abs(right - overlay.bounding_box()["width"]) < 1
            assert not outage["direction_failures"]
            assert len(debug_logs) == 3, debug_logs
            import json
            assert json.loads(debug_logs[0].split("\n", 1)[1])["youtubeId"] == ID
            assert debug_logs[1].startswith("[Soundthing] Conditioned VTT:\nWEBVTT")
            assert expected in debug_logs[1]
            logged_scenes = json.loads(debug_logs[2].split("\n", 1)[1])
            assert logged_scenes[0]["start"] == 0 and logged_scenes[0]["end"] == 10
            assert logged_scenes[0]["env"] == backend.cfg.environments[0].id
            frozen = overlay.locator('[data-wave="border"]').get_attribute("d")
            page.wait_for_timeout(150)
            assert overlay.locator('[data-wave="border"]').get_attribute("d") == frozen
            page.emulate_media(reduced_motion="reduce")
            page.wait_for_timeout(100)
            static = overlay.locator('[data-wave="border"]').get_attribute("d")
            page.evaluate("document.querySelector('video').currentTime=1.7")
            page.wait_for_timeout(100)
            assert overlay.locator('[data-wave="border"]').get_attribute("d") == static
            page.emulate_media(reduced_motion="no-preference")
            raw_track = (backend.RUNS / f"youtube_{ID}" / "captions.conditioned.vtt").read_bytes()
            # Each caption fades from its own VTT end, following the video clock.
            page.evaluate("document.querySelector('video').pause();document.querySelector('video').currentTime=3.375")
            expect(overlay.locator(".caption-line").first).to_have_css("opacity", "0.5")
            assert overlay.locator(".caption-line").count() == 2
            page.wait_for_timeout(150)
            expect(overlay.locator(".caption-line").first).to_have_css("opacity", "0.5")
            page.evaluate("document.querySelector('video').currentTime=1.5")
            expect(overlay.locator(".caption-line")).to_have_count(1)
            expect(overlay.locator(".caption-line")).to_have_css("opacity", "1")
            assert (backend.RUNS / f"youtube_{ID}" / "captions.conditioned.vtt").read_bytes() == raw_track
            print("PASS: independent 0.75-second display tails, media-time fade, pause/seek recovery, and unchanged VTT", flush=True)
            print("PASS: stereo waves, independent recovery from direction 503/404, pause and reduced motion", flush=True)
            page.screenshot(path=str(work / "overlay.png"))
            print("PASS: global toggle, real worker/API/VTT overlay, and closing the popup during processing", flush=True)
            page.evaluate("document.querySelector('video').currentTime=4")
            expect(overlay).to_be_visible()
            expect(overlay.locator("div")).to_be_hidden()
            expect(overlay.locator('[data-wave="border"]')).to_be_visible()
            page.wait_for_timeout(350)
            idle_box = overlay.bounding_box()
            idle_wave = overlay.locator('[data-wave="border"]').get_attribute("d")
            assert overlay.locator("section").inner_text() == ""
            assert overlay.locator("svg path").count() == 1
            left, top, right, bottom = wave_bounds(overlay)
            assert abs(left) < 1 and right > idle_box["width"] + 2
            assert top <= 1 and bottom >= idle_box["height"] - 1
            assert overlay.locator("section").evaluate("el=>getComputedStyle(el).backgroundColor") == "rgba(0, 0, 0, 0)"
            assert overlay.locator('[data-wave="border"]').get_attribute("fill") != "none"
            assert idle_box["width"] >= 191 and idle_box["height"] >= 63
            page.screenshot(path=str(work / "idle-waves.png"))
            page.evaluate("document.querySelector('video').currentTime=2.5")
            expect(overlay.locator("div")).to_contain_text(" · ")
            page.wait_for_timeout(350)
            expanded_box = overlay.bounding_box()
            assert expanded_box["width"] > idle_box["width"] + 50
            assert expanded_box["height"] > idle_box["height"] + 20
            assert overlay.locator('[data-wave="border"]').get_attribute("d") != idle_wave
            view_box = [float(value) for value in overlay.locator("svg").get_attribute("viewBox").split()]
            assert abs(view_box[2] - expanded_box["width"]) < 1
            assert abs(view_box[3] - expanded_box["height"]) < 1
            page.screenshot(path=str(work / "expanded-waves.png"))
            page.evaluate("document.querySelector('video').currentTime=4")
            page.wait_for_timeout(350)
            shrunk_box = overlay.bounding_box()
            assert abs(shrunk_box["width"] - idle_box["width"]) < 1
            assert abs(shrunk_box["height"] - idle_box["height"]) < 1
            print("PASS: always-visible minimum panel, independent waves in caption gaps, and synchronized panel/wave resizing", flush=True)
            page.evaluate("document.querySelector('video').currentTime=2;document.querySelector('video').playbackRate=2")
            expect(overlay.locator("div")).to_contain_text(expected)
            left, top, right, bottom = wave_bounds(overlay)
            assert abs(left) < 1 and right > overlay.bounding_box()["width"] + 2
            page.evaluate("document.querySelector('#movie_player').classList.add('ad-showing')")
            expect(overlay).to_be_hidden()
            page.evaluate("document.querySelector('#movie_player').classList.remove('ad-showing')")
            expect(overlay).to_be_visible()
            page.evaluate("document.querySelector('#movie_player').style.width='1120px'")
            box = overlay.bounding_box()
            parent = page.locator("#movie_player").bounding_box()
            assert box["x"] > parent["x"] + parent["width"] / 2
            assert box["y"] > parent["y"] + parent["height"] / 2
            assert len(debug_logs) == 3, "Seeking and rendering repeated completion logs"
            print("PASS: full VTT and environment scenes appear once in the YouTube console", flush=True)
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
            expect(overlay.locator('[data-wave="border"]')).to_be_visible()
            assert overlay.locator("svg path").count() == 1
            # The caption panel can still be resizing after the track becomes ready.
            deadline = time.monotonic() + 3
            while True:
                left, top, right, bottom = wave_bounds(overlay)
                if top < -2 and bottom > overlay.bounding_box()["height"] + 2:
                    break
                if time.monotonic() >= deadline:
                    break
                page.wait_for_timeout(50)
            assert top < -2 and bottom > overlay.bounding_box()["height"] + 2
            page.screenshot(path=str(work / "surround-waves.png"))
            print("PASS: automatic stereo-to-5.1 switch, front/surround activity combined in one wave border", flush=True)
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
            deadline = time.monotonic() + 10
            while not outage["offline_requests"] and time.monotonic() < deadline:
                page.wait_for_timeout(50)
            assert outage["offline_requests"]
            expect(page.locator('#soundthing-caption-overlay')).to_be_visible()
            assert page.locator('#soundthing-caption-overlay').locator('section').inner_text() == ""
            outage["offline"] = False
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false", timeout=20000)
            restarted = "sS3dE_fG-12"
            outage["missing_status"] = f"/api/videos/youtube_{restarted}"
            page.evaluate(f"history.pushState({{}},'', '/watch?v={restarted}');document.dispatchEvent(new Event('yt-navigate-finish'))")
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false", timeout=20000)
            assert downloads.count(restarted) == 1
            print("PASS: automatic recovery from temporary server outages and missing jobs after restart", flush=True)
            cleared = page.evaluate("async () => soundthing.clearCache()")
            assert cleared["cleared"] and set(cleared["removed"]) == {"results", "downloads"}
            expect(page.locator("#soundthing-caption-overlay")).to_have_count(0)
            page.wait_for_timeout(1100)  # Periodic refresh must not recreate the deleted cache.
            assert not (backend.RUNS / f"youtube_{restarted}").exists()
            assert downloads.count(restarted) == 1
            assert (backend.RUNS / f"youtube_{ID}").is_dir()
            helpers = []
            for context_id, execution in execution_contexts.items():
                if execution.get("auxData", {}).get("isDefault") is not False:
                    continue
                probe = cdp.send("Runtime.evaluate", {"expression": "typeof window.soundthing?.clearCache === 'function'", "contextId": context_id, "returnByValue": True})
                if probe.get("result", {}).get("value") is True:
                    helpers.append(context_id)
            assert helpers, "The helper must also exist in the extension console context"
            result = cdp.send("Runtime.evaluate", {"expression": "soundthing.clearCache()", "contextId": helpers[0], "awaitPromise": True, "returnByValue": True})
            assert result["result"]["value"]["removed"] == []
            sw.evaluate("""async id => {
              const tabs=await chrome.tabs.query({url:'https://www.youtube.com/*'});
              const tab=tabs.find(tab=>new URL(tab.url).searchParams.get('v')===id);
              await chrome.tabs.sendMessage(tab.id,{type:'soundthing:retry'});
            }""", restarted)
            page.wait_for_function("document.querySelector('#soundthing-caption-overlay')?.shadowRoot?.querySelector('div')?.hasAttribute('data-status') === false")
            assert downloads.count(restarted) == 2
            print("PASS: clearing while enabled drops loaded captions/direction and waits for explicit regeneration", flush=True)
            popup = context.new_page()
            popup.goto(f"{origin}/popup.html")
            expect(popup.locator("#enabled")).to_be_checked()
            expect(popup.locator("h1")).to_have_text("soundscape!")
            popup.locator("#enabled").focus()
            popup.keyboard.press("Space")
            expect(popup.locator("#enabled")).not_to_be_checked()
            expect(popup.locator("h1")).to_have_text("soundscape.")
            expect(popup.locator(".surf-caption")).to_be_hidden()
            expect(popup.locator("#click-caption")).to_be_hidden()
            expect(page.locator("#soundthing-caption-overlay")).to_have_count(0)
            expect(second.locator("#soundthing-caption-overlay")).to_have_count(0)
            popup.emulate_media(reduced_motion="reduce")
            assert popup.locator(".shore-wave").first.evaluate("el => el.getAnimations().length") == 0
            popup.locator("body").screenshot(path=str(work / "popup.png"))
            print("PASS: toggle persists and disabling clears every tab", flush=True)
            print("PASS: beach waves move with captions off; shoreline text is enabled-only; reduced motion is respected", flush=True)

            # The normal page console reaches the extension worker, even while off.
            cleared = page.evaluate("async () => soundthing.clearCache()")
            assert cleared["cleared"] and cleared["video_id"] == f"youtube_{restarted}"
            assert set(cleared["removed"]) == {"results", "downloads"}
            assert not (backend.RUNS / f"youtube_{restarted}").exists()
            assert (backend.RUNS / f"youtube_{ID}").is_dir()
            assert page.evaluate("async () => (await soundthing.clearCache()).removed") == []
            wrong_video = page.evaluate("async id => { try { await soundthing.clearCache('https://youtu.be/'+id); } catch (error) { return error.message; } }", OTHER)
            assert "Open that video's" in wrong_video
            api_page = context.new_page()
            response = api_page.goto("http://127.0.0.1:8766")
            assert response.json() == {"name": "soundthing", "status": "ok"}
            assert api_page.evaluate("async id => (await fetch('/api/youtube/cache?url='+id,{method:'DELETE'}).then(r=>r.json())).cleared", OTHER)
            api_page.close()
            assert not (backend.RUNS / f"youtube_{OTHER}").exists()
            assert (backend.RUNS / f"youtube_{ID}").is_dir()
            print("PASS: per-video cache deletion from YouTube and localhost consoles, with other videos preserved", flush=True)

            # Test both developer policies against the same production controller.
            popup.emulate_media(reduced_motion="no-preference")
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
                  if(message.action==='direction')return {ok:true,direction:{duration:5,mode:'stereo',channels:['FL','FR'],frames:[{time:0,levels:{FL:.9,FR:.3}}]}};
                  if(message.action==='scenes')return {ok:true,scenes:[]};
                  if(message.action==='status'){await window.readyGate;return {ok:true,status:'done'}};
                  return {ok:true,vtt:'WEBVTT\\n\\n00:00:00.000 --> 00:00:05.000\\n[test]\\n'};
                }});
                window.policyController.configure({enabled:true,server:'http://localhost:8766'});
              };
            }""", origin)
            popup.evaluate("makePolicyFixture('keep-playing')")
            assert popup.evaluate("pauseCalls") == 0
            popup.wait_for_function("() => policyController.session.waves.track?.mode==='stereo'")
            expect(popup.locator('#soundthing-caption-overlay').locator('[data-wave="border"]')).to_be_visible()
            assert popup.evaluate("policyController.session.status!=='done'")
            assert popup.locator('#soundthing-caption-overlay').locator('section').inner_text() == ""
            # Exercise actual SVG geometry with constant tracks, avoiding layout changes.
            contrast = popup.evaluate("""async () => {
              const {DOA_VISUAL}=await import('./settings.mjs');
              const waves=policyController.session.waves, original=waves.track;
              const sample=(FL,FR,time=0)=>{
                waves.setTrack({duration:5,mode:'stereo',channels:['FL','FR'],frames:[{time:0,levels:{FL,FR}}]});
                waves.render(time,true);
                const points=[...waves.path.getAttribute('d').matchAll(/[ML](-?[\\d.]+),(-?[\\d.]+)/g)].map(p=>({x:+p[1],y:+p[2]}));
                const left=points.reduce((a,b)=>a.x<b.x?a:b),right=points.reduce((a,b)=>a.x>b.x?a:b);
                return {left:-left.x,right:right.x-waves.width,peakY:left.y,height:waves.height};
              };
              const result={quiet:sample(.5,0),loud:sample(.9,0),mixed:sample(.9,.5),symmetric:sample(.9,.9),pulsed:sample(.9,0,1/(4*DOA_VISUAL.swayHz))};
              waves.setTrack(original);policyController.render();
              return result;
            }""")
            assert contrast["loud"]["left"] > contrast["quiet"]["left"] * 4
            assert contrast["mixed"]["left"] > contrast["mixed"]["right"] * 5
            assert contrast["mixed"]["right"] > 1
            assert abs(contrast["symmetric"]["left"] - contrast["symmetric"]["right"]) < .1
            sway = abs(contrast["loud"]["peakY"] - contrast["pulsed"]["peakY"])
            assert 2 < sway < 10
            assert abs(contrast["loud"]["peakY"] - contrast["loud"]["height"] / 2) < 2
            print("PASS: SVG contrast, preserved quieter direction, stereo symmetry, and limited tangential sway", flush=True)
            popup.evaluate("finishPolicy()")
            popup.wait_for_function("() => policyController.state.status==='done'")
            popup.evaluate("""() => {
              const session=policyController.session;
              session.video.currentTime=5;
              Object.defineProperty(session.video,'ended',{get:()=>true});
              session.endedAt=performance.now()-375;
              policyController.render();
            }""")
            opacity = popup.locator('#soundthing-caption-overlay .caption-line').evaluate("el=>+getComputedStyle(el).opacity")
            assert .4 < opacity < .55
            expect(popup.locator('#soundthing-caption-overlay .caption-line')).to_have_count(0)
            print("PASS: the final caption finishes its fade after the video ends", flush=True)
            popup.evaluate("policyController.session.cues=[];policyController.render()")
            expect(popup.locator('#soundthing-caption-overlay')).to_be_visible()
            expect(popup.locator('#soundthing-caption-overlay').locator('div')).to_be_hidden()
            expect(popup.locator('#soundthing-caption-overlay').locator('[data-wave="border"]')).to_be_visible()
            for mode in ("silent", "unavailable"):
                popup.evaluate("""mode=>{
                  const track=mode==='silent'
                    ? {duration:5,mode:'stereo',channels:['FL','FR'],frames:[{time:0,levels:{FL:0,FR:0}}]}
                    : {duration:5,mode:'unavailable',channels:[],frames:[]};
                  policyController.session.waves.setTrack(track);policyController.render();
                }""", mode)
                resting = popup.locator('#soundthing-caption-overlay')
                expect(resting.locator('[data-wave="border"]')).to_be_visible()
                left, top, right, bottom = wave_bounds(resting)
                box = resting.bounding_box()
                assert abs(left) < 1 and abs(top) < 1
                assert abs(right-box["width"]) < 1 and abs(bottom-box["height"]) < 1
                assert resting.locator("svg path").count() == 1
            print("PASS: one shared border/fill, direction before captions, empty VTT, and resting silent/unavailable outline", flush=True)
            popup.evaluate("makePolicyFixture('pause-until-ready')")
            assert popup.evaluate("pausedState && pauseCalls===1")
            popup.evaluate("finishPolicy()")
            popup.wait_for_function("() => policyController.state.status==='done'")
            assert popup.evaluate("!pausedState && playCalls===1")
            popup.evaluate("makePolicyFixture('pause-until-ready')")
            popup.locator("#movie_player").dispatch_event("pointerdown")
            # Synthetic events intentionally do not claim to be user interactions.
            popup.locator("#movie_player").click(force=True)
            popup.evaluate("finishPolicy()")
            popup.wait_for_function("() => policyController.state.status==='done'")
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
