# Soundthing Chrome extension

The toolbar popup controls non-speech captions on standard YouTube watch pages.
Turn Soundthing on once: each video you open is downloaded by the local backend,
processed, and overlaid with the conditioned VTT at the bottom right of the
YouTube player. No Studio handoff, manual MP4 import, or browser download dialog is
required for this flow. Use only your own public or unlisted uploads.

Caption borders also display soundtrack direction waves. No extra toggle or
microphone permission is required; the popup reports the detected audio mode.

## Direction waves

The backend automatically analyzes stereo or verified 5.1 audio from the original
media. Waves follow the **whole soundtrack**, including speech/music, rather than
isolating the event named in a caption. This is speaker-feed activity in the mix,
not a measurement of a recorded object's physical location.

- Stereo: the left and right borders follow their respective channel levels.
  Equal channels show symmetric activity without implying a front/rear location.
- 5.1: the box is a top-down compass, with forward above and behind below. Center
  is 0°, front left/right ±30°, and surround left/right ±110°. Wave anchors are
  the intersections of those rays with the actual box border, including after
  resizing. Top/bottom indicate front/behind, never height. Both FFmpeg `5.1`
  and `5.1(side)` layouts use these nominal playback positions.
- LFE (bass) is excluded from direction. Mono and unidentified layouts have no
  direction waves. A stereo mix is never upmixed to fabricate surround channels.
- Waves appear while a caption is visible, follow seeking/playback speed, freeze
  on pause, and disappear during ads or when Soundthing is off. Reduced-motion
  preferences replace oscillation with static channel-strength lines.

Downloads prefer available six-channel audio in the selected soundtrack language,
otherwise stereo. FFprobe verifies the downloaded layout. YouTube may expose only
stereo to the downloader, even for an upload originally made with surround sound.
The downloaded soundtrack may differ from an alternate language/audio track you
manually select in YouTube; the extension does not detect those selections.

Analysis streams the original media in 50 ms RMS windows; -60..0 dBFS maps to 0..1,
with 50 ms attack and 200 ms release smoothing. The existing mono model input and
stereo modifier input remain unchanged. Results are cached in
`runs/<video_id>/direction.json`, independently of caption events. An unavailable
result is also cached; remove only this artifact to request another analysis.

Older completed runs gain direction data when the extension requests it, without
regenerating captions. The server may reacquire the original into
`uploads/youtube/<youtube_id>/channel-preserved/` to check for surround audio;
earlier downloads remain intact. This backfill shares the serialized YouTube job
queue. If reacquisition fails, the existing original is analyzed instead.
Direction failures never hide working captions. The extension retries missing
direction tracks independently and keeps ready tracks in tab memory.

## Setup

From the repository root, in WSL/Linux:

```sh
bash extension/dev.sh setup
bash extension/dev.sh doctor
```

`setup` creates `.venv` if needed and installs the Python server, test, and yt-dlp
dependencies. It doesn't install system packages or change shell profiles.
`doctor` reports missing tools and checks an explicitly configured cookie file
without displaying its contents. These commands also work when the script is
invoked by its absolute path from another directory.

Install `ffmpeg` and `ffprobe` on the server PATH, for example with
`sudo apt install ffmpeg` on Ubuntu. Install a supported Node.js runtime too
(Node 24 works); the downloader explicitly selects yt-dlp's Node runtime. The
`default` yt-dlp extra supplies its JavaScript challenge components. Start the
server without manually activating the virtual environment:

```sh
bash extension/dev.sh serve
```

`soundthing serve` and `doctor` also find tools installed beside their Python
interpreter, including our local `.venv/bin/ffmpeg` and `ffprobe` installation.
For authenticated YouTube downloads, use `bash extension/dev.sh serve --cookies
"/absolute/path/to/cookies.txt"`; see the authentication steps below.

The default server is `http://127.0.0.1:8000`. It must stay running; closing the
popup does not stop it. Server and model configuration remain in the existing
pipeline setup.

In Chrome:

1. Open `chrome://extensions`, enable **Developer mode**, select **Load unpacked**,
   and choose this `extension/` folder (the folder containing `manifest.json`).
2. Pin Soundthing, open one of your YouTube videos, and click its toolbar icon.
3. Turn on **Soundthing**. The preference applies to all standard YouTube tabs.
4. Wait for queued → downloading → processing → ready. The popup may be closed.
5. Captions synchronize to the video's current time, including seeking and speed
   changes. YouTube's existing captions remain enabled and untouched.

**Windows Chrome + WSL:** copy the entire extension folder to a Windows directory,
then load that copy. In this workspace the installed copy is
`C:\Users\ailen\Downloads\soundthing-extension`. After code changes, run:

```sh
bash extension/dev.sh sync
```

The helper asks Windows for its user profile and updates its
`Downloads/soundthing-extension` folder. For another destination, use
`bash extension/dev.sh sync --destination "/mnt/c/Users/YOUR_NAME/path/to/extension"`.
It refuses to copy into the source folder, its parents, or a nonempty folder that
isn't a Soundthing installation. It copies runtime assets and documentation,
leaving tests and Python caches out of the installation and preserving unrelated
existing files. Click **Reload** on the extension card, then refresh your YouTube
tabs so the new content script loads. There is no frontend build step.

The equivalent Python CLI commands are `soundthing doctor` and
`soundthing sync-extension --destination PATH`.

## YouTube authentication

If the popup says YouTube requires browser authentication, you can explicitly
configure an exported cookie file. This works across Windows Chrome and WSL
without having the Linux process decrypt Chrome's Windows cookie database.
Soundthing does not read browser cookies automatically or request cookie access.

Follow [yt-dlp's YouTube cookie-export guidance](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies):

1. Use a Netscape-format exporter listed in
   [yt-dlp's FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp),
   such as **Get cookies.txt LOCALLY** for Chrome. If using an incognito window,
   explicitly enable the exporter there in Chrome's extension settings.
2. Open one incognito tab, sign in to YouTube, then navigate that same tab to
   `https://www.youtube.com/robots.txt`.
3. Export the `youtube.com` cookies, then close that incognito window. Export only
   the YouTube session rather than all your browser cookies.
4. Store the file outside the repository, for example in
   `C:\Users\ailen\Downloads\youtube.cookies.txt`. Its WSL path is
   `/mnt/c/Users/ailen/Downloads/youtube.cookies.txt`. Keep it private; don't paste
   its contents into chat or commit it.
5. Stop the existing server with Ctrl+C, then restart it from the repository root:

   ```sh
   bash extension/dev.sh serve --cookies "/mnt/c/Users/ailen/Downloads/youtube.cookies.txt"
   ```

   Use the actual filename from your exporter. In this workspace the saved file
   is `www.youtube.com_cookies.txt`. The `--cookies` option is equivalent to setting
   `SOUNDTHING_YOUTUBE_COOKIES` and validates the file before starting the server.

6. Click **Retry this video** in the Soundthing popup. No extension reload or
   new Chrome permissions are required for cookie-file support.

The configured file must be readable and begin with `# Netscape HTTP Cookie File`
or `# HTTP Cookie File`. Both metadata and media requests use a private temporary
copy (file mode 0600 inside a 0700 directory), which is removed on success or
failure. yt-dlp may update that copy; the original export is never changed.
Cookie data is never returned through the API, placed in extension storage, or
included in authenticated downloader error messages.

When the file expires, export it again and retry. If its path changes, restart
the server with the new environment variable. To stop using cookies, unset
`SOUNDTHING_YOUTUBE_COOKIES`, omit `--cookies`, and restart the server. Personal
yt-dlp config files remain ignored; only the explicit environment variable or
`serve --cookies` option configures authentication.
Cookies may help with the challenge but don't guarantee access; persistent
blocking needs further diagnosis. Account access through yt-dlp also carries
the limitations described in its linked guidance.

## Merge sound captions with speech subtitles

Merge an existing generated sound track with a UTF-8 SRT or VTT speech track,
without downloading the video again or running models:

```sh
.venv/bin/soundthing merge-captions \
  runs/youtube_lpQPfGkmkr8/captions.conditioned.vtt \
  /path/to/speech.srt \
  --out /path/to/combined.vtt
```

Use an `.srt` output filename to export SubRip instead. The output must differ
from both inputs. Cue text and timestamps are read from either input format;
UTF-8 BOMs, Windows line endings, multiline cues, VTT cue identifiers and settings,
and VTT comments/style blocks are supported. Layout metadata, cue positioning,
regions, and stylesheet rules are not retained in the combined track.

During overlaps, speech appears first and sound captions appear on separate
lines in the same cue. The merger splits the timeline at cue boundaries instead
of shifting either track. Adjacent identical cues coalesce, and identical
simultaneous text appears once. Empty intervals stay empty. Invalid cue timing or
encoding is reported rather than silently skipping cues.

You can also request merged tracks while running one video or one inputs file:

```sh
source .venv/bin/activate
soundthing run /path/to/video.mp4 --speech /path/to/speech.srt
```

This adds `captions.baseline.merged.vtt` and `captions.conditioned.merged.vtt` beside
the standard sound-only VTT files. Event JSON and canonical sound-only captions
are preserved. Speech subtitles are not transcribed or fetched automatically;
you supply the existing subtitle file. This merge is currently a CLI feature;
the YouTube overlay continues displaying sound-only cues alongside YouTube's own
speech captions, avoiding duplicate dialogue. Merged tracks can be opened as
subtitle files in your video player.

## Controls and behavior

- The global toggle starts off and persists across browser sessions. Turning off
  immediately removes overlays and stops new requests. Submitted backend jobs
  finish and remain cached; this does not cancel other tabs' jobs.
- The popup shows status, errors, Retry on failure, and a collapsible local-server
  connection setting. It identifies demo captions when either model backend is a
  stub. Actual caption quality depends on the existing model owners' backends.
- Temporary connection failures, timeouts, HTTP 408/429, and server errors enter
  **Reconnecting** and retry automatically with backoff up to 15 seconds. A missing
  job after restart is resubmitted. Terminal downloader/model errors still require
  Retry. Turning off cancels browser retries; completed captions already loaded in
  a tab continue working even if the server stops. A connection failure releases
  an extension-owned pause rather than leaving playback stuck.
- Standard watch pages, theater mode, and player fullscreen are supported. Shorts,
  embedded players, live/upcoming streams, and private-video support
  are excluded. Captions are hidden during YouTube ads.
- Duplicate videos share a backend job. Completed artifacts are reused, including
  after a server restart; failed jobs can be retried. The server processes YouTube
  jobs one at a time.
- The downloader prefers up to 480p video with audio for environment classification;
  if no capped format exists it falls back to an available video. Media stays in
  `uploads/youtube/<youtube_id>/`; pipeline artifacts live in
  `runs/youtube_<youtube_id>/`. There is no automatic cleanup or cache expiry.
- After changing model configuration, stop the server and remove the specific
  generated `runs/youtube_<youtube_id>/` cache directory to regenerate that video's
  captions. This extension does not change shared model backends or configuration.
- Download failures (including YouTube sign-in/bot challenges) are shown in the
  popup. Cookies are used only when an exported file is explicitly configured
  on the backend as described above. Updating yt-dlp may be needed
  when YouTube changes: `python -m pip install -U 'yt-dlp[default]'`.
- **Open video workspace** preserves the original file-upload library, playback,
  and original/VTT downloads as a secondary workflow. Its server setting remains
  independent of the popup setting.

## Developer playback setting

In `settings.mjs`, set `PLAYBACK_POLICY` to one of:

- `"keep-playing"` (default): let playback continue; show cues for the current time
  once the track is ready.
- `"pause-until-ready"`: pause during preparation and resume when ready only if
  Soundthing paused a playing video and the user has not intervened. Videos already
  paused by the user stay paused. Disabling or a failed job releases an
  extension-owned pause; navigating to a different video does not resume it.

Reload the extension and refresh YouTube pages after changing this constant.
Both policies use the same production controller and are exercised by browser
smoke tests.

## Backend API and extension state

Person 3's additive HTTP API:

- `POST /api/youtube` with `{"youtube_id":"<11-character-id>"}` returns HTTP 202
  and `{"video_id":"youtube_<id>"}`. It reuses active/complete jobs and retries
  failed ones. Only canonical YouTube IDs are accepted; no caller-controlled URLs,
  output paths, cookies, or download commands.
- `GET /api/videos/{video_id}` returns `status` (`queued`, `downloading`,
  `processing`, `done`, `error`), optional `error`, and `demo` (whether either
  configured model backend is a stub). Only complete runs are recovered as done
  after restart; interrupted jobs can be resubmitted.
- `GET /api/videos/{video_id}/captions/conditioned.vtt` supplies the existing VTT.
- `GET /api/videos/{video_id}/direction` supplies a validated bare `DirectionTrack`
  (`duration`, `mode`, `channels`, `frames`); 404 while absent. A request for an
  older complete run queues its missing analysis. Frames contain media `time` and
  normalized channel `levels`. See `docs/interfaces.md` for the approved contract.

The additive direction contract uses the existing schema version; event,
environment, extraction, and stage interfaces are unchanged. Job status is in
memory; generated files use the documented contracts.
The server is a single-process local demo, not a multi-user job service.

`chrome.storage.local.soundthingPreferences` holds `{enabled, server}`. The old
`soundthingWorkspace` library metadata is preserved separately. Video bytes and
VTT tracks are not saved in extension storage. Content-script state is ephemeral:
reopening a video asks the server to reuse any completed job.

The background worker validates the extension sender, top frame, current YouTube
watch ID, and enabled preference before making narrowly constructed local API
requests. Long work belongs to Python, not Chrome's temporary popup or worker.
Polling and overlays belong to each content script. The overlay uses text content
and isolated shadow styles rather than interpreting captions as HTML.

## Permissions

- `storage`: persistent global preference, server address, and the secondary library.
- `downloads`: save originals/VTT files from the secondary workspace only.
- `http://127.0.0.1/*` and `http://localhost/*`: local API access.
- `https://www.youtube.com/*`: watch-page detection and isolated content script.

No wildcard internet permission, Google account access, cookie permission, or
server CORS changes. Chrome 120 or newer is required.

## Verification

With the virtual environment active and Node.js 20+:

```sh
node --test --test-isolation=none extension/tests/*.test.mjs
pytest -q
soundthing check-config
```

The `--test-isolation=none` flag requires Node.js 22+; on Node 20, omit that flag.
Optional Chromium integration check:

```sh
python -m pip install playwright
python -m playwright install chromium
python extension/tests/browser_smoke.py
```

This runs the real extension, background worker, API, and VTT writer with a
controlled downloader, processor, YouTube DOM and generated native media clock.
It covers closing the popup mid-job, seeking/speed changes, ad suppression, SPA
navigation, cross-tab deduplication, global disabling, both playback policies,
fullscreen containment, page refreshes, replaced video elements, temporary server
outages, and missing-job recovery after restart.
It makes no live YouTube download and doesn't verify current YouTube DOM selectors.

For a live check, open a team-owned public/unlisted video, enable Soundthing, verify
both caption files on the server, seek to a cue, and check the bottom-right overlay
in normal, theater, and fullscreen modes. Turn off and confirm it disappears.
