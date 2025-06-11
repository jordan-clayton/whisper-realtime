# Ribble-Whisper

An adapter library that provides a high-level API for realtime (streaming) and offline transcription using
OpenAI's Whisper. This library wraps core [whisper-rs](https://github.com/tazz4843/whisper-rs) functionality,
voice activity detection (VAD), file loading, as well as additional optional features (e.g. resampling) to make
integrating Whisper into projects fast and easy.

Realtime transcription is fairly difficult to run well without GPU acceleration.
It is recommended to build this crate using one of the exposed GPU backends (CUDA, Vulkan, etc.) to achieve
better performance (see: [Features](#features)). Depending on your hardware, it might be feasible to run realtime on
CPU using a smaller quantized model (e.g. base.en-q5, tiny.en-q5). These are likely to be less accurate, but
your results may vary.

***NOTE: While this API is being used in [Ribble](https://github.com/jordan-clayton/ribble), bear in mind that it is a
young project and may not necessarily be a good fit for production.***

## External Dependencies

- At this time, Ribble-Whisper does not support the use of multiple audio input backends. You will require SDL2 to build
  this library.
- If you are building with an optional GPU backend (e.g. CUDA), you will require the appropriate SDK

## Building

```bash
git clone --recursive https://github.com/jordan-clayton/ribble-whisper.git
cd ribble_whisper
cargo build --release 
# To build with a GPU backend or one of the optional features, run:
# cargo build --release --features <Desired Features>
```

Building is expected to work out-of-the-box for Windows/macOS/Linux.
If you encounter issues with building due to whisper-rs,
check [here](https://github.com/tazz4843/whisper-rs/blob/master/BUILDING.md).

## Stream Example

```bash
git clone --recursive https://github.com/jordan-clayton/ribble-whisper.git
cd ribble_whisper
# Hardware acceleration is more-or-less required for realtime transcription.
# Swap out CUDA with your preferred backend (see: Features).
cargo run --example realtime_stream --features "cuda downloader"
```

## Example Usage

```rust
// TODO: Snippet from realtime_stream + explanation
```

- From: [realtime_stream](https://github.com/jordan-clayton/whisper-realtime/blob/main/examples/realtime_stream.rs)

## Features

### Whisper Backends:

- TODO

### Symphonia Codecs:

- TODO

### Additional Features:

- TODO

## License

This project is licensed under MIT. See [LICENSE](./LICENSE).

## Third-Party Licenses

This project includes third-party software components:

- [`symphonia`](https://github.com/pdeljanov/symphonia) (Mozilla Public License 2.0)
    - Used for audio decoding and file loading.
    - MPL-2.0 is a file-level copyleft license. This does **not** affect your usage of this library unless
      you modify and redistribute Symphonia source files.
    - See [`LICENSE-MPL-2.0`](./LICENSE-MPL-2.0) for the full license.
- [`earshot`](https://github.com/pykeio/earshot) (BSD-3-Clause)
    - Used for (fallback) voice activity detection
    - See [`LICENSE-BSD-3`](./LICENSE-BSD-3) for the full license.

## A note about tests

Many of the tests and benchmarks in this project rely on audio files not included as part of the project.
These will need to be replaced if you wish to run any of the tests. I do not claim any rights to these files
nor will they ever be distributed; they are for testing purposes only.