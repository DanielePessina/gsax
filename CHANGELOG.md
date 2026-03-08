# Changelog

## 0.4.0

- Add `SamplingResult.save()` for serializing samples + metadata to disk
- Add `gsax.load()` for reconstructing `SamplingResult` from saved files
- Supported sample formats: csv, txt, xlsx, parquet, pkl
- Storage-optimized: identity mappings skip the .npz sidecar
