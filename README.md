# Whisper Realtime

A rust-based, realtime-capable, wrapper library for [whisper-rs](https://github.com/tazz4843/whisper-rs).
Seeks to provide an ergonomic API to support realtime and static speech transcription applications.

At this time, accurate realtime transcription (and use of larger models) requires GPU acceleration.
Offline transcription is possible with larger models using the CPU.

See: Cargo.toml for feature flags to compile with GPU support

- See examples for usage

### Note: This is not production-ready and the API may change at any time during development.

### Full documentation is coming (hopefully) soon to accompany a somewhat stable API.

## TODO:

- Finish README
- Documentation
- Feature flags Documentation

## License

This project is licensed under MIT. See [LICENSE](./LICENSE).

### Third-Party Licenses

This project includes third-party software components:

- [`symphonia`](https://github.com/pdeljanov/symphonia) (Mozilla Public License 2.0)
    - Used for audio decoding and file loading.
    - MPL-2.0 is a file-level copyleft license. This does **not** affect your usage of this library unless you modify
      and redistribute Symphonia source files.
    - See [`LICENSE-MPL-2.0`](./LICENSE-MPL-2.0) for the full license.
- [`earshot`](https://github.com/pykeio/earshot) (BSD-3-Clause)
    - Used for (fallback) voice activity detection
    - See [`LICENSE-BSD-3`](./LICENSE-BSD-3) for the full license.

### Tests

Many of the tests and benchmarks in this project rely on audio files not included as part of the project.
These will need to be replaced if you wish to run any of the tests. I do not claim any rights to these files
nor will they ever be distributed; they are for testing purposes only.