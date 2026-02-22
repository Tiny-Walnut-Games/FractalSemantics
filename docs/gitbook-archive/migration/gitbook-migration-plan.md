# Migration Plan: Future GitBook Space

This archive is intentionally GitBook-friendly.

## What is already prepared

- `SUMMARY.md` navigation map
- Topic-oriented markdown pages
- Stable file/folder structure under `docs/gitbook-archive/`

## Migration steps when GitBook space exists

1. Create GitBook space.
2. Import `docs/gitbook-archive/` content.
3. Set `SUMMARY.md` as navigation source.
4. Validate links and command formatting.
5. Keep this folder as backup/source, or switch to bidirectional sync.

## Recommended ownership model

- Source of truth in repo for versioned docs.
- GitBook as publishing layer.
- Release checklist includes docs sync verification.

## Optional future enhancements

- Add changelog page for public CLI/API changes.
- Add architecture diagrams for runner/analysis flow.
- Add troubleshooting matrix by experiment ID.
