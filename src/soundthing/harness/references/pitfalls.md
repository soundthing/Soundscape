# Pitfalls

Traps that aren't obvious from the code. Add one when something bites; remove it when the
cause is gone.

- **AudioSet audio is a mirror, not Google's.** Google publishes only labels. Audio comes from Hugging Face `agkphysics/AudioSet` (eval split, ~24 GB of parquet); its `video_id` lacks the `_<start_ms>` suffix of strong-label segment ids, and ~3% of mapped clips are missing.
- **Holdout clips must never appear in dev.** AudioSet holdout clips evaluate models
  strictly on promotion; allowing holdout ids into dev risks tuning against the test split,
  so import refuses overlaps and enforces disjoint sets.
- **`import kaggle` authenticates at import time** and raises without credentials, so the harness never imports it; it shells out to the CLI.
- **New Kaggle tokens (`KGAT_…`) need kaggle>=1.8, which needs Python 3.11+.** The project venv is 3.10, so the CLI lives in its own tool env (`uv tool install --python 3.12 kaggle`); the token goes in `~/.kaggle/access_token`, not `kaggle.json`.
- **`kaggle datasets create -p <relative/path>` fails on Windows** (it builds an upload-cache filename from the path). Run it from inside the folder with `-p .`.
- **Kaggle output is everything in `/kaggle/working`.** The kernel keeps harness state in `/tmp` and copies out only caches, dev labels and meta, so holdout labels and audio never come back.
- **Faking a harness submodule needs the package attribute too.** Once the real module has
  been imported, `import soundthing.harness.x as m` resolves through the package
  attribute, so a fake placed only in `sys.modules` is silently bypassed (tests pass alone,
  fail in the full suite). Patch both.
- **MCP 2.x server import path.** MCP 2.x renamed `FastMCP` to `mcp.server.mcpserver.MCPServer`; importing from `mcp.server.fastmcp` fails. Use `from mcp.server.mcpserver import MCPServer`.
- **YAML line endings may be CRLF.** Config YAML files checked out on Windows may have CRLF line endings, so line edits must preserve the existing newline style to avoid corrupting git diffs.
- **Cue matching must pair with event class IDs, not parse caption text.** Rendered cues do not contain raw class IDs, and different classes may map to identical or modifier-altered caption text. The harness pairs cues with sorted events to retain ground-truth class IDs without text parsing.

