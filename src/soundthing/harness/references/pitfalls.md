# Pitfalls

Traps that aren't obvious from the code. Add one when something bites; remove it when the
cause is gone.

- **AudioSet audio is a mirror, not Google's.** Google publishes only labels. Audio comes from Hugging Face `agkphysics/AudioSet` (eval split, ~24 GB of parquet); its `video_id` lacks the `_<start_ms>` suffix of strong-label segment ids, and ~3% of mapped clips are missing.
- **Holdout clips must never appear in dev.** AudioSet holdout clips evaluate models
  strictly on promotion; allowing holdout ids into dev risks tuning against the test split,
  so import refuses overlaps and enforces disjoint sets.
- **`import kaggle` authenticates at import time.** Importing `kaggle` or any of its submodules without valid credentials immediately raises; import lazily inside client accessors only.
- **Kaggle builds only committed and pushed code.** The cache kernel clones the repo at a specified git SHA; uncommitted local edits or unpushed commits will not be in the remote build.
- **Faking a harness submodule needs the package attribute too.** Once the real module has
  been imported, `import soundthing.harness.x as m` resolves through the package
  attribute, so a fake placed only in `sys.modules` is silently bypassed (tests pass alone,
  fail in the full suite). Patch both.
- **MCP 2.x server import path.** MCP 2.x renamed `FastMCP` to `mcp.server.mcpserver.MCPServer`; importing from `mcp.server.fastmcp` fails. Use `from mcp.server.mcpserver import MCPServer`.
- **YAML line endings may be CRLF.** Config YAML files checked out on Windows may have CRLF line endings, so line edits must preserve the existing newline style to avoid corrupting git diffs.
