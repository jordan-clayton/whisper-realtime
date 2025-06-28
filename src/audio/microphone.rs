use crate::audio::recorder::{Recorder, SampleSink};
#[cfg(feature = "sdl2")]
use sdl2::audio::AudioDevice;
#[cfg(feature = "sdl2")]
use sdl2::audio::AudioFormat;

// TODO: document/rename
// NOTE: this is just a quick little adapter for SDL's AudioFormat, filtering out all
// unsupported audio formats (by type + endianness).
// It is unlikely that a system will record in a format that doesn't match its endianness,
// but afaik it's not a guarantee.

// This is to be used mainly for getting information about whether to build a fallback device.
#[derive(Copy, Clone, PartialEq)]
pub enum RibbleAudioFormat {
    F32,
    I16,
    Invalid,
}

impl RibbleAudioFormat {
    pub fn is_invalid(&self) -> bool {
        matches!(self, RibbleAudioFormat::Invalid)
    }
}

#[cfg(feature = "sdl2")]
impl From<AudioFormat> for RibbleAudioFormat {
    fn from(value: AudioFormat) -> Self {
        let little_endian = cfg!(target_endian = "little");
        match (value, little_endian) {
            (AudioFormat::F32LSB, true) => RibbleAudioFormat::F32,
            (AudioFormat::F32LSB, false) => RibbleAudioFormat::Invalid,
            (AudioFormat::F32MSB, false) => RibbleAudioFormat::F32,
            (AudioFormat::F32MSB, true) => RibbleAudioFormat::Invalid,
            (AudioFormat::S16LSB, true) => RibbleAudioFormat::I16,
            (AudioFormat::S16LSB, false) => RibbleAudioFormat::Invalid,
            (AudioFormat::S16MSB, false) => RibbleAudioFormat::I16,
            (AudioFormat::S16MSB, true) => RibbleAudioFormat::Invalid,
            _ => RibbleAudioFormat::Invalid,
        }
    }
}

/// Trait for starting/stopping audio capture.
pub trait MicCapture {
    fn play(&self);
    fn pause(&self);
    fn sample_rate(&self) -> usize;
    fn format(&self) -> RibbleAudioFormat;
    fn channels(&self) -> u8;
    fn buffer_size(&self) -> usize;
}

#[cfg(feature = "sdl2")]
pub struct Sdl2Capture<S: SampleSink> {
    device: AudioDevice<Recorder<S>>,
}

impl<S: SampleSink> Sdl2Capture<S> {
    pub fn new(device: AudioDevice<Recorder<S>>) -> Self {
        Self { device }
    }
}

impl<S: SampleSink> MicCapture for Sdl2Capture<S> {
    fn play(&self) {
        self.device.resume()
    }
    fn pause(&self) {
        self.device.pause()
    }
    fn sample_rate(&self) -> usize {
        self.device.spec().freq as usize
    }
    fn format(&self) -> RibbleAudioFormat {
        self.device.spec().format.into()
    }
    fn channels(&self) -> u8 {
        self.device.spec().channels
    }
    fn buffer_size(&self) -> usize {
        self.device.spec().samples as usize
    }
}

// Eventual TODO: other backends
// e.g. CpalCapture...
