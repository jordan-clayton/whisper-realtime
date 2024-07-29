# Whisper Realtime

A rust-based, realtime capable, wrapper library for [whisper-rs](https://github.com/tazz4843/whisper-rs).
Seeks to provide an ergonomic API to support realtime and static speech transcription applications.

At this time, accurate realtime transcription (and use of larger models) requires GPU acceleration.
Static transcription is possible with larger models via the CPU.

See: Cargo.toml for feature flags to compile with GPU support (proper documentation coming)

- See examples for use.

## TODO:

- RealtimeTranscriber: Non VAD implementation
- Finish readme
- Proper documentation