"""Person 3: targeted URL-based cache deletion and job coordination."""
import importlib

import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from soundthing.server.youtube import CacheBusyError, YouTubeJobs, youtube_id_from_url
from .test_youtube_server import ID, OTHER_ID, artifacts, make_source


@pytest.mark.parametrize('url', [
    ID, f'https://www.youtube.com/watch?v={ID}&t=15', f'https://youtu.be/{ID}?si=test',
    f'https://m.youtube.com/watch?v={ID}', f'https://youtube.com/shorts/{ID}',
    f'https://www.youtube.com/embed/{ID}', f'https://youtube.com/live/{ID}',
    f'https://studio.youtube.com/video/{ID}/edit',
])
def test_supported_urls_resolve_to_one_video(url):
    assert youtube_id_from_url(url) == ID


@pytest.mark.parametrize('url', [
    '', '../outside', f'https://evil.example/watch?v={ID}',
    f'https://youtube.com.evil.example/watch?v={ID}', f'https://youtube.com@evil.example/{ID}',
    f'file:///youtube.com/watch?v={ID}', 'https://youtube.com/watch?v=../outside',
    f'https://youtube.com:8000/watch?v={ID}', f'https://user@youtube.com/watch?v={ID}',
    f'https://youtube.com/watch/other?v={ID}',
])
def test_invalid_urls_cannot_select_cache_paths(url):
    with pytest.raises(ValueError):
        youtube_id_from_url(url)


@pytest.fixture
def jobs(tmp_path, cfg):
    instance = YouTubeJobs(tmp_path / 'runs', tmp_path / 'uploads', cfg, {},
        downloader=lambda video_id, uploads: make_source(uploads, video_id), processor=artifacts)
    yield instance
    instance.close()


def test_clear_removes_only_selected_video_and_next_submission_reprocesses(jobs):
    for video_id in (ID, OTHER_ID):
        source = make_source(jobs.uploads, video_id)
        artifacts(source, jobs.runs, jobs.cfg)
        download = jobs.uploads / 'youtube' / video_id / 'channel-preserved' / 'selection'
        download.mkdir(parents=True)
        (download / 'video.mkv').write_bytes(b'download')
        jobs.status[source.stem] = {'status': 'done'}
    (jobs.uploads / 'cookies.txt').write_text('preserve')
    result = jobs.clear_cache(ID)
    assert result == {'video_id': f'youtube_{ID}', 'cleared': True, 'removed': ['results', 'downloads']}
    assert not (jobs.runs / f'youtube_{ID}').exists()
    assert not (jobs.uploads / 'youtube' / ID).exists()
    assert not (jobs.uploads / f'youtube_{ID}.mp4').exists()
    assert f'youtube_{ID}' not in jobs.status
    assert (jobs.runs / f'youtube_{OTHER_ID}').exists()
    assert (jobs.uploads / 'youtube' / OTHER_ID).exists()
    assert jobs.status[f'youtube_{OTHER_ID}'] == {'status': 'done'}
    assert (jobs.uploads / 'cookies.txt').read_text() == 'preserve'
    assert jobs.clear_cache(ID)['removed'] == []
    jobs.submit(ID)
    jobs.executor.submit(lambda: None).result(timeout=3)
    assert jobs.status[f'youtube_{ID}']['status'] == 'done'


@pytest.mark.parametrize('state', ['queued', 'downloading', 'processing', 'direction'])
def test_busy_video_is_not_deleted(jobs, state):
    source = make_source(jobs.uploads, ID)
    artifacts(source, jobs.runs, jobs.cfg)
    if state == 'direction':
        jobs.direction_pending.add(source.stem)
    else:
        jobs.status[source.stem] = {'status': state}
    with pytest.raises(CacheBusyError):
        jobs.clear_cache(ID)
    assert source.exists()
    assert (jobs.runs / source.stem / 'inputs.json').exists()


def test_cache_symlink_is_unlinked_without_touching_its_target(jobs, tmp_path):
    outside = tmp_path / 'original'
    outside.mkdir()
    (outside / 'keep').write_text('original')
    jobs.runs.mkdir()
    (jobs.runs / f'youtube_{ID}').symlink_to(outside, target_is_directory=True)
    assert jobs.clear_cache(ID)['removed'] == ['results']
    assert (outside / 'keep').read_text() == 'original'


def test_download_parent_cannot_redirect_deletion_outside_uploads(jobs, tmp_path):
    outside = tmp_path / 'original'
    (outside / ID).mkdir(parents=True)
    jobs.uploads.mkdir()
    (jobs.uploads / 'youtube').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='outside'):
        jobs.clear_cache(ID)
    assert (outside / ID).is_dir()


def test_api_clears_url_reports_busy_and_rejects_invalid_input(jobs, monkeypatch):
    app = importlib.import_module('soundthing.server.app')
    monkeypatch.setattr(app, 'youtube_jobs', jobs)
    monkeypatch.setattr(app, 'RUNS', jobs.runs)
    monkeypatch.setattr(app, 'status', jobs.status)
    source = make_source(jobs.uploads, ID)
    artifacts(source, jobs.runs, jobs.cfg)
    jobs.status[source.stem] = {'status': 'processing'}
    params = {'url': f'https://www.youtube.com/watch?v={ID}'}
    with TestClient(app.app) as client:
        assert client.delete('/api/youtube/cache', params=params).status_code == 409
        jobs.status[source.stem] = {'status': 'done'}
        response = client.delete('/api/youtube/cache', params=params)
        assert response.status_code == 200 and response.json()['cleared'] is True
        assert client.get(f'/api/videos/{source.stem}').status_code == 404
        assert client.get(f'/api/videos/{source.stem}/captions/conditioned.vtt').status_code == 404
        assert client.delete('/api/youtube/cache', params=params).json()['removed'] == []
        assert client.delete('/api/youtube/cache', params={'url': 'https://evil.example/'}).status_code == 400
        assert client.delete('/api/youtube/cache').status_code == 422
