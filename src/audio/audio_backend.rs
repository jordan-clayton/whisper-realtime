use crate::utils::constants::{AUDIO_BUFFER_SIZE, WHISPER_SAMPLE_RATE};

use crate::audio::microphone::{MicCapture, Sdl2Capture};
use crate::audio::recorder::{Recorder, SampleSink};
use crate::utils::errors::RibbleWhisperError;

#[cfg(feature = "sdl2")]
use sdl2::AudioSubsystem;
#[cfg(feature = "sdl2")]
use sdl2::audio::AudioSpecDesired;

/// Encapsulates required recording spec information.
/// Set fields to None to use device defaults.
pub struct CaptureSpec {
    /// The sample rate (in Hz).
    sample_rate: Option<usize>,
    /// The number of channels.
    channels: Option<u8>,
    /// The size-limit (in bytes) before triggering the audio callback to fire. Must be a
    /// power of 2
    period: Option<usize>,
}

impl CaptureSpec {
    pub fn new() -> Self {
        Self {
            sample_rate: None,
            channels: None,
            period: None,
        }
    }
    pub fn with_sample_rate(self, sample_rate: Option<usize>) -> Self {
        Self {
            sample_rate,
            channels: self.channels,
            period: self.period,
        }
    }

    pub fn with_num_channels(self, num_channels: Option<u8>) -> Self {
        Self {
            sample_rate: self.sample_rate,
            channels: num_channels,
            period: self.period,
        }
    }

    pub fn with_period(self, period: Option<usize>) -> Self {
        Self {
            sample_rate: self.sample_rate,
            channels: self.channels,
            period,
        }
    }

    pub fn sample_rate(&self) -> Option<usize> {
        self.sample_rate
    }
    pub fn channels(&self) -> Option<u8> {
        self.channels
    }
    pub fn period(&self) -> Option<usize> {
        self.period
    }
}

#[cfg(feature = "sdl2")]
impl From<AudioSpecDesired> for CaptureSpec {
    fn from(value: AudioSpecDesired) -> Self {
        let AudioSpecDesired {
            freq,
            channels,
            samples,
        } = value;

        let sample_rate = freq.map(|f| f as usize);
        let period = samples.map(|sam| sam as usize);

        Self::new()
            .with_sample_rate(sample_rate)
            .with_num_channels(channels)
            .with_period(period)
    }
}

#[cfg(feature = "sdl2")]
impl From<CaptureSpec> for AudioSpecDesired {
    fn from(value: CaptureSpec) -> Self {
        let freq = value.sample_rate().map(|freq| freq as i32);
        let channels = value.channels();
        let samples = value.period().map(|samples| samples as u16);
        Self {
            freq,
            channels,
            samples,
        }
    }
}

impl Default for CaptureSpec {
    fn default() -> Self {
        Self::new()
            .with_sample_rate(Some(WHISPER_SAMPLE_RATE as usize))
            .with_num_channels(Some(1))
            .with_period(Some(AUDIO_BUFFER_SIZE))
    }
}

pub trait AudioBackend<S: SampleSink>: Sized {
    type Capture: MicCapture;
    fn open_capture(&self, spec: CaptureSpec, sink: S)
    -> Result<Self::Capture, RibbleWhisperError>;
}

#[cfg(feature = "sdl2")]
/// The default audio backend. Can be integrated with Sdl2 by using [SdlBackend::from_subsystem].
/// ***Note: SDL2 is not thread-safe, so SdlBackends cannot be safely shared across threads.***
/// However, audio devices created using SdlBackend::open_capture() are guaranteed to be
/// thread-safe and can be shared freely. It is left up to the implementation to handle managing
/// SDL2 lifetimes and quirks.
pub struct Sdl2Backend {
    audio_subsystem: AudioSubsystem,
}

#[cfg(feature = "sdl2")]
impl Sdl2Backend {
    /// Constructs an SdlBackend using the provided AudioSubsystem
    pub fn from_subsystem(audio_subsystem: AudioSubsystem) -> Self {
        Self { audio_subsystem }
    }
}

#[cfg(feature = "sdl2")]
impl<S: SampleSink> AudioBackend<S> for Sdl2Backend {
    type Capture = Sdl2Capture<S>;

    fn open_capture(
        &self,
        spec: CaptureSpec,
        sink: S,
    ) -> Result<Self::Capture, RibbleWhisperError> {
        let valid_period = spec.period().is_none_or(|period| period.is_power_of_two());

        if !valid_period {
            return Err(RibbleWhisperError::DeviceError(format!(
                "Invalid Audio Device period size: {:?}",
                spec.period()
            )));
        }

        let audio_spec: AudioSpecDesired = spec.into();
        let device = self
            .audio_subsystem
            .open_capture(None, &audio_spec, |_| Recorder::new(sink))
            .map_err(|e| {
                RibbleWhisperError::DeviceError(format!("Failed to build audio capture: {e}"))
            })?;

        Ok(Sdl2Capture::new(device))
    }
}

#[cfg(feature = "sdl2")]
/// Convenience function that handles initializing SDL and an [SdlBackend] for obtaining a capture
/// device. If managing SDL2 independenently, construct using `Sdlbackend::from_subsystem()`
/// See: [SdlBackend] for information about thread-safety.
pub fn default_backend() -> Result<(sdl2::Sdl, Sdl2Backend), RibbleWhisperError> {
    let ctx = sdl2::init()
        .map_err(|e| RibbleWhisperError::DeviceError(format!("Failed to open SDL context: {e}")))?;
    let audio_subsystem = ctx.audio().map_err(|e| {
        RibbleWhisperError::DeviceError(format!("Failed to open Sdl2 AudioSubsystem: {e}"))
    })?;

    let backend = Sdl2Backend::from_subsystem(audio_subsystem);

    Ok((ctx, backend))
}
