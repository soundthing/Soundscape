import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
const root = new URL('../', import.meta.url);

test('popup and content scripts have scoped permissions and all local assets exist', async () => {
  const manifest = JSON.parse(await readFile(new URL('manifest.json', root), 'utf8'));
  assert.equal(manifest.manifest_version, 3);
  assert.deepEqual(manifest.permissions, ['storage']);
  assert.deepEqual(manifest.host_permissions, ['http://127.0.0.1/*', 'http://localhost/*', 'https://www.youtube.com/*']);
  assert.equal(manifest.action.default_popup, 'popup.html');
  assert.deepEqual(manifest.content_scripts[0].matches, ['https://www.youtube.com/*']);
  assert.match(manifest.content_security_policy.extension_pages, /script-src 'self'/);
  const paths = [manifest.background.service_worker, ...manifest.content_scripts.flatMap(script => script.js), ...manifest.web_accessible_resources[0].resources,
    ...Object.values(manifest.icons), ...Object.values(manifest.action.default_icon)];
  for (const path of paths) await access(new URL(path, root));
  for (const entry of [manifest.action.default_popup]) {
    const html = await readFile(new URL(entry, root), 'utf8');
    for (const [, path] of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
      if (!path.startsWith('https:')) await access(new URL(path, root));
    }
    assert.doesNotMatch(html, /<script[^>]*>\s*[^<\s]/);
  }
});

test('page-accessible modules expose all of their static relative imports', async () => {
  const manifest = JSON.parse(await readFile(new URL('manifest.json', root), 'utf8'));
  const exposed = new Set(manifest.web_accessible_resources.flatMap(entry => entry.resources));
  for (const path of exposed) {
    if (!/\.(?:mjs|js)$/.test(path)) continue;
    const moduleUrl = new URL(path, root);
    const source = await readFile(moduleUrl, 'utf8');
    for (const [, dependency] of source.matchAll(/(?:import|export)\s+[^;]*?\bfrom\s*["'](\.[^"']+)["']/g)) {
      const relative = new URL(dependency, moduleUrl).href.slice(root.href.length);
      assert.ok(exposed.has(relative), `${path} imports ${relative}, which Chrome must expose to the page`);
    }
  }
});
