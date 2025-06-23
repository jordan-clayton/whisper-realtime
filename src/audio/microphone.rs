use std::sync::Arc;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::audio::recorder::{
    AudioInputAdapter, ClosedLoopRecorder, FallbackClosedLoopRecorder, FallbackFanoutRecorder,
    FallbackUseArc, FallbackUseVec, FanoutRecorder, RecorderSample, StereoMonoConverter,
    StereoMonoConvertible, UseArc, UseVec,
};
use crate::utils::constants::{CONVERT_MONO_TO_STEREO, CONVERT_STEREO_TO_MONO};
use crate::utils::errors::RibbleWhisperError;
use crate::utils::{constants, Sender};
use enum_dispatch::enum_dispatch;
use sdl2::audio::{AudioCallback, AudioDevice, AudioFormat, AudioSpecDesired};
use sdl2::{AudioSubsystem, Sdl};

/// A Basic Audio Backend that uses SDL to gain access to the default audio input
/// At this time there is no support for other audio backends, but this may change in the future.
#[derive(Clone)]
pub struct AudioBackend {
    sdl_ctx: Arc<Sdl>,
    audio_subsystem: AudioSubsystem,
}

impl AudioBackend {
    /// Initializes the audio backend to access the microphone when running a realtime transcription
    /// It is not encouraged to call new more than once, but it is not an error to do so.
    /// Returns an error if the backend fails to initialize.
    pub fn new() -> Result<Self, RibbleWhisperError> {
        let ctx = sdl2::init().map_err(|e| {
            RibbleWhisperError::ParameterError(format!(
                "Failed to create SDL context, error: {}",
                e
            ))
        })?;

        let audio_subsystem = ctx.audio().map_err(|e| {
            RibbleWhisperError::ParameterError(format!(
                "Failed to initialize audio subsystem, error: {}",
                e
            ))
        })?;

        let sdl_ctx = Arc::new(ctx);
        Ok(Self {
            sdl_ctx,
            audio_subsystem,
        })
    }

    /// To access the inner [Sdl] context
    pub fn sdl_ctx(&self) -> Arc<Sdl> {
        self.sdl_ctx.clone()
    }
    /// To access the inner [AudioSubsystem]
    pub fn audio_subsystem(&self) -> AudioSubsystem {
        self.audio_subsystem.clone()
    }

    /// A convenience method that prepares [AudioDevice] for use
    /// with [crate::transcriber::realtime_transcriber::RealtimeTranscriber] to transcribe
    /// audio realtime. Use the fanout capture when doing other audio processing concurrently
    /// with transcription.
    ///
    /// # Arguments:
    /// * audio_sender: a message sender to forward audio from the input device
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(RibbleWhisperError) on failure to build.
    /// See: [MicCaptureBuilder] for error conditions.
    pub fn build_whisper_fanout_default<AC: AudioInputAdapter + Send + Clone>(
        &self,
        audio_sender: Sender<AC::SenderOutput>,
    ) -> Result<FanoutMicCapture<AC>, RibbleWhisperError> {
        self.build_whisper_default().build_fanout(audio_sender)
    }

    /// A convenience method that prepares [AudioDevice] for use
    /// with [crate::transcriber::realtime_transcriber::RealtimeTranscriber] to transcribe
    /// audio realtime. Use the closed loop capture when only transcription processing is required.
    ///
    /// # Arguments:
    /// * buffer: a ringbuffer for storing audio from the input device.
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(RibbleWhisperError) on failure to build.
    /// See: [MicCaptureBuilder] for error conditions.
    pub fn build_whisper_closed_loop_default<T: RecorderSample>(
        &self,
        buffer: &AudioRingBuffer<T>,
    ) -> Result<ClosedLoopMicCapture<T>, RibbleWhisperError> {
        self.build_whisper_default().build_closed_loop(buffer)
    }

    pub fn build_whisper_default(&self) -> MicCaptureBuilder {
        self.build_microphone()
            .with_num_channels(Some(1))
            .with_sample_rate(Some(constants::WHISPER_SAMPLE_RATE as i32))
            .with_sample_size(Some(constants::AUDIO_BUFFER_SIZE))
    }

    /// Returns a builder to set up the audio capture parameters
    pub fn build_microphone(&self) -> MicCaptureBuilder {
        MicCaptureBuilder::new(&self.audio_subsystem)
    }
}

struct ProbeCapture;
impl AudioCallback for ProbeCapture {
    type Channel = u8;
    fn callback(&mut self, _: &mut [Self::Channel]) {}
}

/// A builder for setting (SDL) audio input configurations
#[derive(Clone)]
pub struct MicCaptureBuilder<'a> {
    audio_subsystem: &'a AudioSubsystem,
    audio_spec_desired: AudioSpecDesired,
}

impl<'a> MicCaptureBuilder<'a> {
    pub fn new(audio_subsystem: &'a AudioSubsystem) -> Self {
        // AudioSpecDesired does not implement Default
        let audio_spec_desired = AudioSpecDesired {
            freq: None,
            channels: None,
            samples: None,
        };

        Self {
            audio_subsystem,
            audio_spec_desired,
        }
    }
    /// To change the [AudioSubsystem]
    pub fn with_audio_subsystem(mut self, audio_subsystem: &'a AudioSubsystem) -> Self {
        self.audio_subsystem = audio_subsystem;
        self
    }

    /// To change the desired sample rate
    pub fn with_sample_rate(mut self, sample_rate: Option<i32>) -> Self {
        self.audio_spec_desired.freq = sample_rate;
        self
    }

    /// To change the desired number of channels (e.g. 1 = mono, 2 = stereo)
    /// NOTE: Realtime transcription requires audio to be in/converted to mono
    /// [crate::transcriber::realtime_transcriber::RealtimeTranscriber] does not handle conversion.
    pub fn with_num_channels(mut self, num_channels: Option<u8>) -> Self {
        self.audio_spec_desired.channels = num_channels;
        self
    }

    /// To set the input audio buffer size.
    /// NOTE: this must be a power of two.
    /// Providing an invalid size will result in falling back to default settings
    pub fn with_sample_size(mut self, samples: Option<u16>) -> Self {
        self.audio_spec_desired.samples = samples.filter(|s| s.is_power_of_two());
        self
    }

    /// To set the desired audio spec all at once.
    /// This will not be useful unless you are already managing SDL on your own.
    ///
    pub fn with_desired_audio_spec(mut self, spec: AudioSpecDesired) -> Self {
        self.audio_spec_desired = spec;
        self
    }

    /// Builds [AudioDevice] to open audio capture (e.g. for use in realtime transcription).
    /// Fans out data via message passing for use when doing additional audio processing concurrently
    /// with transcription.
    /// # Arguments:
    /// * audio_sender: a message sender to forward audio from the input device
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(RibbleWhisperError) on an SDL failure.
    pub fn build_fanout<AC: AudioInputAdapter + Send + Clone>(
        self,
        sender: Sender<AC::SenderOutput>,
    ) -> Result<FanoutMicCapture<AC>, RibbleWhisperError> {
        let device = self
            .audio_subsystem
            .open_capture(None, &self.audio_spec_desired, |_| {
                FanoutRecorder::new(sender)
            })
            .map_err(|e| {
                RibbleWhisperError::ParameterError(format!("Failed to build audio stream: {}", e))
            })?;
        Ok(FanoutMicCapture { device })
    }

    // TODO: document -> this method doesn't consume;
    // it only provides a diff against what the system returned
    pub fn probe_audio_device(
        &self,
        format: RibbleAudioFormat,
    ) -> Result<AudioDeviceDiff, RibbleWhisperError> {
        if let RibbleAudioFormat::Invalid = format {
            return Err(RibbleWhisperError::ParameterError(
                "Must provide valid audio format: i16/f32. Received invalid.".to_string(),
            ));
        }

        let probe = self
            .audio_subsystem
            .open_capture(None, &self.audio_spec_desired, |_| ProbeCapture)
            .map_err(|e| {
                RibbleWhisperError::ParameterError(format!("Failed to build audio probe: {}", e))
            })?;

        let probe_spec = probe.spec();

        let desired = &self.audio_spec_desired;
        let diff = AudioDeviceDiff {
            format: AudioParameterDiff {
                want: format,
                got: probe_spec.format.into(),
            },
            channels: AudioParameterDiff {
                want: desired.channels.unwrap_or_else(|| probe_spec.channels),
                got: probe_spec.channels,
            },
            sample_rate: AudioParameterDiff {
                want: desired.freq.unwrap_or_else(|| probe_spec.freq),
                got: probe_spec.freq,
            },
            buffer_size: AudioParameterDiff {
                want: desired.samples.unwrap_or_else(|| probe_spec.samples),
                got: probe_spec.samples,
            },
        };
        drop(probe);
        Ok(diff)
    }
    // TODO: this probably needs some tlc, but the basic idea should follow.
    // It's probably nooot going to do what I'm thinking (re type), but it's a start
    // NOTE: it must be true that AC::SenderOutput = needed_specs.format.want,
    // And AC::SenderInput = needed_specs.format.got,
    // NOTE: AC::SenderOutput must match the desired output type.
    // TODO: these also require testing.

    // Handles both I16 -> I16 and F32 -> I16 conversions
    // Builds using an Arc sender
    // Fails if the wrong datatypes are requested via needed_specs.
    // want = convert_to, got = convert_from
    pub fn build_fanout_fallback_arc_int(
        self,
        sender: Sender<Arc<[i16]>>,
        needed_specs: AudioDeviceDiff,
    ) -> Result<FallbackMic, RibbleWhisperError> {
        let (convert_from, convert_to, converter_bitmask, buffer_len) =
            self.get_fallback_mic_params(needed_specs)?;
        match (convert_from, convert_to) {
            (RibbleAudioFormat::F32, RibbleAudioFormat::I16) => {
                let stereo_to_mono: StereoMonoConverter<f32> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;

                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackFanoutRecorder::new_arc_int(sender, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::F32ToS16FanoutArc(FallbackFanoutMicCapture {
                    device,
                }))
            }
            (RibbleAudioFormat::I16, RibbleAudioFormat::I16) => {
                let stereo_to_mono: StereoMonoConverter<i16> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;

                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackFanoutRecorder::new_arc_int(sender, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::S16ToS16FanoutArc(FallbackFanoutMicCapture {
                    device,
                }))
            }
            (_, _) => Err(RibbleWhisperError::DeviceCompatibilityError(
                "Audio input device does not support required format".to_string(),
            )),
        }
    }

    // Handles both F32 -> F32 and I16 -> F32 conversions
    // Builds using an Arc sender
    // Fails if the wrong datatypes are requested via needed_specs.
    // want = convert_to, got = convert_from
    pub fn build_fanout_fallback_arc_float(
        self,
        sender: Sender<Arc<[f32]>>,
        needed_specs: AudioDeviceDiff,
    ) -> Result<FallbackMic, RibbleWhisperError> {
        let (convert_from, convert_to, converter_bitmask, buffer_len) =
            self.get_fallback_mic_params(needed_specs)?;
        match (convert_from, convert_to) {
            (RibbleAudioFormat::F32, RibbleAudioFormat::F32) => {
                let stereo_to_mono: StereoMonoConverter<f32> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;
                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackFanoutRecorder::new_arc_fp(sender, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::F32ToF32FanoutArc(FallbackFanoutMicCapture {
                    device,
                }))
            }

            (RibbleAudioFormat::I16, RibbleAudioFormat::F32) => {
                let stereo_to_mono: StereoMonoConverter<i16> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;

                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackFanoutRecorder::new_arc_fp(sender, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::S16ToF32FanoutArc(FallbackFanoutMicCapture {
                    device,
                }))
            }
            (_, _) => Err(RibbleWhisperError::DeviceCompatibilityError(
                "Audio input device does not support required format".to_string(),
            )),
        }
    }

    // Handles both I16 -> I16 and F32 -> I16 conversions
    // Builds using a vec sender
    // Fails if the wrong datatypes are requested via needed_specs.
    // want = convert_to, got = convert_from
    pub fn build_fanout_fallback_vec_int(
        self,
        sender: Sender<Vec<i16>>,
        needed_specs: AudioDeviceDiff,
    ) -> Result<FallbackMic, RibbleWhisperError> {
        let (convert_from, convert_to, converter_bitmask, buffer_len) =
            self.get_fallback_mic_params(needed_specs)?;
        match (convert_from, convert_to) {
            (RibbleAudioFormat::F32, RibbleAudioFormat::I16) => {
                let stereo_to_mono: StereoMonoConverter<f32> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;
                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackFanoutRecorder::new_vec_int(sender, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::F32ToS16FanoutVec(FallbackFanoutMicCapture {
                    device,
                }))
            }
            (RibbleAudioFormat::I16, RibbleAudioFormat::I16) => {
                let stereo_to_mono: StereoMonoConverter<i16> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;

                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackFanoutRecorder::new_vec_int(sender, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::S16ToS16FanoutVec(FallbackFanoutMicCapture {
                    device,
                }))
            }
            (_, _) => Err(RibbleWhisperError::DeviceCompatibilityError(
                "Audio input device does not support required format".to_string(),
            )),
        }
    }

    // Handles both I16 -> F32 and F32 -> F32 conversions
    // Builds using a vec sender
    // Fails if the wrong datatypes are requested via needed_specs.
    // want = convert_to, got = convert_from
    pub fn build_fanout_fallback_vec_float(
        self,
        sender: Sender<Vec<f32>>,
        needed_specs: AudioDeviceDiff,
    ) -> Result<FallbackMic, RibbleWhisperError> {
        let (convert_from, convert_to, converter_bitmask, buffer_len) =
            self.get_fallback_mic_params(needed_specs)?;
        match (convert_from, convert_to) {
            (RibbleAudioFormat::F32, RibbleAudioFormat::F32) => {
                let stereo_to_mono: StereoMonoConverter<f32> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;
                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackFanoutRecorder::new_vec_fp(sender, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::F32ToF32FanoutVec(FallbackFanoutMicCapture {
                    device,
                }))
            }
            (RibbleAudioFormat::I16, RibbleAudioFormat::F32) => {
                let stereo_to_mono: StereoMonoConverter<i16> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;

                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackFanoutRecorder::new_vec_fp(sender, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;

                Ok(FallbackMic::S16ToF32FanoutVec(FallbackFanoutMicCapture {
                    device,
                }))
            }

            (_, _) => Err(RibbleWhisperError::DeviceCompatibilityError(
                "Audio input device does not support required format".to_string(),
            )),
        }
    }
    fn get_fallback_mic_params(
        &self,
        needed_specs: AudioDeviceDiff,
    ) -> Result<(RibbleAudioFormat, RibbleAudioFormat, u8, usize), RibbleWhisperError> {
        // These two are non-negotiable
        if !needed_specs.sample_rate().matches() {
            // TODO: this needs to return a DeviceError
            return Err(RibbleWhisperError::DeviceCompatibilityError(
                "Audio device does not support desired sample rate.".to_string(),
            ));
        }

        // Possibly don't include this one? It doesn't really matter if the input buffer size
        // isn't what's requested...
        if !needed_specs.buffer_size().matches() {
            return Err(RibbleWhisperError::DeviceCompatibilityError(
                "Audio device does not support input buffer size.".to_string(),
            ));
        }

        // The rest can be handled by fallback
        let converter_bitmask = match needed_specs.channels {
            AudioParameterDiff { want: 1, got: 1 } => 0,
            AudioParameterDiff { want: 1, got: 2 } => CONVERT_MONO_TO_STEREO,
            AudioParameterDiff { want: 2, got: 2 } => 0,
            AudioParameterDiff { want: 2, got: 1 } => CONVERT_STEREO_TO_MONO,
            _ => {
                return Err(RibbleWhisperError::DeviceCompatibilityError(
                    "Invalid number of channels desired, or device is not supported".to_string(),
                ));
            }
        };
        // If it's going mono to stereo, the buffer_len doubles
        // If it's going stereo to mono, the
        let buffer_len = if converter_bitmask != 0 {
            (needed_specs.buffer_size().got as f32
                * match converter_bitmask {
                    CONVERT_MONO_TO_STEREO => 2f32,
                    CONVERT_STEREO_TO_MONO => 0.5,
                    _ => 1f32,
                }) as usize
        } else {
            needed_specs.buffer_size().got as usize
        };

        let AudioParameterDiff {
            want: convert_to,
            got: convert_from,
        } = needed_specs.format;
        Ok((convert_from, convert_to, converter_bitmask, buffer_len))
    }

    /// Builds [AudioDevice] to open audio capture (e.g. for use in realtime transcription).
    /// Writes directly into the ringbuffer, for when only transcription is required.
    /// Prefer the fanout implementation when doing additional processing during transcription
    /// to guarantee data coherence.
    ///
    /// # Arguments:
    /// * buffer: a ringbuffer for storing audio from the input device.
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(RibbleWhisperError) on an SDL failure.
    pub fn build_closed_loop<T: RecorderSample>(
        self,
        buffer: &AudioRingBuffer<T>,
    ) -> Result<ClosedLoopMicCapture<T>, RibbleWhisperError> {
        let device = self
            .audio_subsystem
            .open_capture(None, &self.audio_spec_desired, |_| {
                ClosedLoopRecorder::new(buffer.clone())
            })
            .map_err(|e| {
                RibbleWhisperError::ParameterError(format!("Failed to build audio stream: {}", e))
            })?;
        Ok(ClosedLoopMicCapture { device })
    }

    pub fn build_fallback_closed_loop_int(
        self,
        ring_buffer: AudioRingBuffer<i16>,
        needed_specs: AudioDeviceDiff,
    ) -> Result<FallbackMic, RibbleWhisperError> {
        let (convert_from, convert_to, converter_bitmask, buffer_len) =
            self.get_fallback_mic_params(needed_specs)?;
        match (convert_from, convert_to) {
            (RibbleAudioFormat::F32, RibbleAudioFormat::I16) => {
                let stereo_to_mono: StereoMonoConverter<f32> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;
                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackClosedLoopRecorder::new(ring_buffer, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::F32ToS16ClosedLoop(
                    FallbackClosedLoopMicCapture { device },
                ))
            }
            (RibbleAudioFormat::I16, RibbleAudioFormat::I16) => {
                let stereo_to_mono: StereoMonoConverter<i16> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;

                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackClosedLoopRecorder::new(ring_buffer, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::S16ToS16ClosedLoop(
                    FallbackClosedLoopMicCapture { device },
                ))
            }
            (_, _) => Err(RibbleWhisperError::DeviceCompatibilityError(
                "Audio input device does not support required format".to_string(),
            )),
        }
    }
    pub fn build_fallback_closed_loop_float(
        self,

        ring_buffer: AudioRingBuffer<f32>,
        needed_specs: AudioDeviceDiff,
    ) -> Result<FallbackMic, RibbleWhisperError> {
        let (convert_from, convert_to, converter_bitmask, buffer_len) =
            self.get_fallback_mic_params(needed_specs)?;
        match (convert_from, convert_to) {
            (RibbleAudioFormat::F32, RibbleAudioFormat::F32) => {
                let stereo_to_mono: StereoMonoConverter<f32> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;
                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackClosedLoopRecorder::new(ring_buffer, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;
                Ok(FallbackMic::F32ToF32ClosedLoop(
                    FallbackClosedLoopMicCapture { device },
                ))
            }
            (RibbleAudioFormat::I16, RibbleAudioFormat::F32) => {
                let stereo_to_mono: StereoMonoConverter<i16> =
                    StereoMonoConverter::new(buffer_len).with_conversions(converter_bitmask)?;

                let device = self
                    .audio_subsystem
                    .open_capture(None, &self.audio_spec_desired, |_| {
                        FallbackClosedLoopRecorder::new(ring_buffer, stereo_to_mono)
                    })
                    .map_err(|e| {
                        RibbleWhisperError::ParameterError(format!(
                            "Failed to build audio stream: {}",
                            e
                        ))
                    })?;

                Ok(FallbackMic::S16ToF32ClosedLoop(
                    FallbackClosedLoopMicCapture { device },
                ))
            }
            (_, _) => Err(RibbleWhisperError::DeviceCompatibilityError(
                "Audio input device does not support required format".to_string(),
            )),
        }
    }
}

// TODO: document/place somewhere relevant, possibly cull visibility
#[derive(Clone)]
pub struct AudioParameterDiff<T> {
    want: T,
    got: T,
}

impl<T: PartialEq> AudioParameterDiff<T> {
    pub fn matches(&self) -> bool {
        self.want == self.got
    }
    pub fn borrow_unpack(&self) -> (&T, &T) {
        (&self.want, &self.got)
    }
    pub fn unpack(self) -> (T, T) {
        (self.want, self.got)
    }
}

impl<T: Copy> Copy for AudioParameterDiff<T> {}

// TODO: document/possibly rename, place somewhere relevant, possibly cull visibility
#[derive(Copy, Clone)]
pub struct AudioDeviceDiff {
    format: AudioParameterDiff<RibbleAudioFormat>,
    channels: AudioParameterDiff<u8>,
    sample_rate: AudioParameterDiff<i32>,
    buffer_size: AudioParameterDiff<u16>,
}

// TODO: Document whole thing
// NOTE: this is an Answer and an Ask
// If returned from probing, this holds the information diffs between what the system will give
// when attempting to build the Mic

// Channels + Format can be handled, Sample rate cannot (Resampling needs to happen offline)
// When trying to build a fallback, Copy/Mutate the original AudioDeviceDiff, and use the builder
// to set fields that are absolutely non-negotiable.
// TODO: might be better to use a semantic struct I dunno?
impl AudioDeviceDiff {
    // TODO: document -> this is an "A-Ok, you're going to get what you ask for" function
    pub fn matches(&self) -> bool {
        let format_matches = self.format.matches();
        let channels_matches = self.channels.matches();
        let samples_matches = self.sample_rate.matches();
        format_matches && channels_matches && samples_matches
    }

    // Since these are all cheap copies, there's no need to borrow
    pub fn format(&self) -> AudioParameterDiff<RibbleAudioFormat> {
        self.format
    }
    pub fn channels(&self) -> AudioParameterDiff<u8> {
        self.channels
    }
    pub fn sample_rate(&self) -> AudioParameterDiff<i32> {
        self.sample_rate
    }
    pub fn buffer_size(&self) -> AudioParameterDiff<u16> {
        self.buffer_size
    }
    pub fn with_format_need(mut self, format: RibbleAudioFormat) -> Self {
        self.format.want = format;
        self
    }
    pub fn with_channels_need(mut self, channels: u8) -> Self {
        self.channels.want = channels;
        self
    }
    pub fn with_sample_rate_need(mut self, sample_rate: i32) -> Self {
        self.sample_rate.want = sample_rate;
        self
    }
    pub fn with_buffer_size_need(mut self, buffer_size: u16) -> Self {
        self.buffer_size.want = buffer_size;
        self
    }
}

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
    fn is_invalid(&self) -> bool {
        matches!(self, RibbleAudioFormat::Invalid)
    }
}

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

// TODO: figure out how to get this down to a Zero-Cost Abstraction to avoid fat pointers
// Future TODO: look at macros/dynamically building this at runtime
// TODO: test this.

pub enum MicHandle<T: MicCapture> {
    Requested(T),
    Fallback(FallbackMic),
}
impl<T: MicCapture> MicCapture for MicHandle<T> {
    fn play(&self) {
        match self {
            MicHandle::Requested(m) => m.play(),
            MicHandle::Fallback(m) => m.play(),
        }
    }

    fn pause(&self) {
        match self {
            MicHandle::Requested(m) => m.pause(),
            MicHandle::Fallback(m) => m.pause(),
        }
    }

    fn sample_rate(&self) -> i32 {
        match self {
            MicHandle::Requested(m) => m.sample_rate(),
            MicHandle::Fallback(m) => m.sample_rate(),
        }
    }

    fn format(&self) -> RibbleAudioFormat {
        match self {
            MicHandle::Requested(m) => m.format(),
            MicHandle::Fallback(m) => m.format(),
        }
    }

    fn channels(&self) -> u8 {
        match self {
            MicHandle::Requested(m) => m.channels(),
            MicHandle::Fallback(m) => m.channels(),
        }
    }

    fn buffer_size(&self) -> u16 {
        match self {
            MicHandle::Requested(m) => m.buffer_size(),
            MicHandle::Fallback(m) => m.buffer_size(),
        }
    }
}

#[enum_dispatch(MicCapture)]
pub enum FallbackMic {
    // Fan out
    F32ToF32FanoutArc(FallbackFanoutMicCapture<FallbackUseArc<f32, f32>, StereoMonoConverter<f32>>),
    F32ToF32FanoutVec(FallbackFanoutMicCapture<FallbackUseVec<f32, f32>, StereoMonoConverter<f32>>),
    F32ToS16FanoutArc(FallbackFanoutMicCapture<FallbackUseArc<f32, i16>, StereoMonoConverter<f32>>),
    F32ToS16FanoutVec(FallbackFanoutMicCapture<FallbackUseVec<f32, i16>, StereoMonoConverter<f32>>),
    S16ToS16FanoutArc(FallbackFanoutMicCapture<FallbackUseArc<i16, i16>, StereoMonoConverter<i16>>),
    S16ToS16FanoutVec(FallbackFanoutMicCapture<FallbackUseVec<i16, i16>, StereoMonoConverter<i16>>),
    S16ToF32FanoutArc(FallbackFanoutMicCapture<FallbackUseArc<i16, f32>, StereoMonoConverter<i16>>),
    S16ToF32FanoutVec(FallbackFanoutMicCapture<FallbackUseVec<i16, f32>, StereoMonoConverter<i16>>),
    // Closed Loop
    F32ToF32ClosedLoop(FallbackClosedLoopMicCapture<f32, f32, StereoMonoConverter<f32>>),
    F32ToS16ClosedLoop(FallbackClosedLoopMicCapture<f32, i16, StereoMonoConverter<f32>>),
    S16ToS16ClosedLoop(FallbackClosedLoopMicCapture<i16, i16, StereoMonoConverter<i16>>),
    S16ToF32ClosedLoop(FallbackClosedLoopMicCapture<i16, f32, StereoMonoConverter<i16>>),
}

/// Trait for starting/stopping audio capture.
#[enum_dispatch]
pub trait MicCapture {
    fn play(&self);
    fn pause(&self);
    fn sample_rate(&self) -> i32;
    // TODO: wrapper struct to encapsulate SDL formats
    fn format(&self) -> RibbleAudioFormat;
    fn channels(&self) -> u8;
    fn buffer_size(&self) -> u16;
}

/// Encapsulates [AudioDevice] and sends audio samples out via message channels.
/// Use when performing other audio processing concurrently with transcription
/// (see: examples/realtime_stream.rs).
/// Due to the use of SDL, this cannot be shared across threads.
pub struct FanoutMicCapture<AC>
where
    AC: AudioInputAdapter + Clone + Send,
{
    device: AudioDevice<FanoutRecorder<AC>>,
}

impl<AC> MicCapture for FanoutMicCapture<AC>
where
    AC: AudioInputAdapter + Clone + Send,
{
    fn play(&self) {
        self.device.resume()
    }
    fn pause(&self) {
        self.device.pause()
    }
    fn sample_rate(&self) -> i32 {
        self.device.spec().freq
    }
    // TODO: wrapper struct to encapsulate SDL formats
    fn format(&self) -> RibbleAudioFormat {
        self.device.spec().format.into()
    }
    fn channels(&self) -> u8 {
        self.device.spec().channels
    }
    fn buffer_size(&self) -> u16 {
        self.device.spec().samples
    }
}
pub struct FallbackFanoutMicCapture<AC, S>
where
    AC: AudioInputAdapter + Clone + Send,
    S: StereoMonoConvertible<AC::SenderInput> + Send,
{
    device: AudioDevice<FallbackFanoutRecorder<AC, S>>,
}

impl<AC: AudioInputAdapter + Clone + Send, S: StereoMonoConvertible<AC::SenderInput> + Send>
    MicCapture for FallbackFanoutMicCapture<AC, S>
{
    fn play(&self) {
        self.device.resume()
    }

    fn pause(&self) {
        self.device.pause()
    }

    fn sample_rate(&self) -> i32 {
        self.device.spec().freq
    }

    fn format(&self) -> RibbleAudioFormat {
        self.device.spec().format.into()
    }

    fn channels(&self) -> u8 {
        self.device.spec().channels
    }

    fn buffer_size(&self) -> u16 {
        self.device.spec().samples
    }
}

/// Encapsulates [AudioDevice] and writes directly into [AudioRingBuffer].
/// Use when only transcription processing is required.
/// Due to the use of SDL, this cannot be shared across threads.
pub struct ClosedLoopMicCapture<T: RecorderSample> {
    device: AudioDevice<ClosedLoopRecorder<T>>,
}

impl<T: RecorderSample> MicCapture for ClosedLoopMicCapture<T> {
    fn play(&self) {
        self.device.resume()
    }

    fn pause(&self) {
        self.device.pause()
    }

    fn sample_rate(&self) -> i32 {
        self.device.spec().freq
    }

    fn format(&self) -> RibbleAudioFormat {
        self.device.spec().format.into()
    }

    fn channels(&self) -> u8 {
        self.device.spec().channels
    }

    fn buffer_size(&self) -> u16 {
        self.device.spec().samples
    }
}
pub struct FallbackClosedLoopMicCapture<T1, T2, S>
where
    T1: RecorderSample,
    T2: RecorderSample,
    S: StereoMonoConvertible<T1>,
    FallbackClosedLoopRecorder<T1, T2, S>: AudioCallback,
{
    device: AudioDevice<FallbackClosedLoopRecorder<T1, T2, S>>,
}

impl<T1, T2, S> MicCapture for FallbackClosedLoopMicCapture<T1, T2, S>
where
    T1: RecorderSample,
    T2: RecorderSample,
    S: StereoMonoConvertible<T1>,
    FallbackClosedLoopRecorder<T1, T2, S>: AudioCallback,
{
    fn play(&self) {
        self.device.resume()
    }

    fn pause(&self) {
        self.device.pause()
    }

    fn sample_rate(&self) -> i32 {
        self.device.spec().freq
    }

    fn format(&self) -> RibbleAudioFormat {
        self.device.spec().format.into()
    }

    fn channels(&self) -> u8 {
        self.device.spec().channels
    }

    fn buffer_size(&self) -> u16 {
        self.device.spec().samples
    }
}

// The following functions are exposed but their use is not encouraged unless required.

/// This is deprecated and will be removed at a later date.
/// Prefer [MicCaptureBuilder].
#[inline]
pub fn get_desired_audio_spec(
    freq: Option<i32>,
    channels: Option<u8>,
    samples: Option<u16>,
) -> AudioSpecDesired {
    AudioSpecDesired {
        freq,
        channels,
        samples,
    }
}

/// Visibility is currently exposed due to legacy implementations.
/// Do not rely on this function, as it will be marked private in the future.
/// Prefer [MicCaptureBuilder]
#[inline]
pub fn build_audio_stream<AC: AudioInputAdapter + Send>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<AC::SenderOutput>,
) -> Result<AudioDevice<FanoutRecorder<AC>>, RibbleWhisperError> {
    let audio_stream = audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| FanoutRecorder::new(audio_sender),
        )
        .map_err(|e| {
            RibbleWhisperError::ParameterError(format!("Failed to build audio stream: {}", e))
        })?;

    Ok(audio_stream)
}

/// Visibility is currently exposed due to legacy implementations.
/// Do not rely on this function, as it will be marked private in the future.
/// Prefer [MicCaptureBuilder]
#[inline]
pub fn build_audio_stream_vec_sender<T: RecorderSample>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<Vec<T>>,
) -> Result<AudioDevice<FanoutRecorder<UseVec<T>>>, RibbleWhisperError> {
    let audio_stream = audio_subsystem.open_capture(
        // Device - should be default, SDL should change if the user changes devices in their sysprefs.
        None,
        desired_spec,
        |_spec| FanoutRecorder::new_vec(audio_sender),
    );
    match audio_stream {
        Err(e) => Err(RibbleWhisperError::ParameterError(format!(
            "Failed to build audio stream: {}",
            e
        ))),
        Ok(audio) => Ok(audio),
    }
}

/// Visibility is currently exposed due to legacy implementations.
/// Do not rely on this function, as it will be marked private in the future.
/// Prefer [MicCaptureBuilder]
#[inline]
pub fn build_audio_stream_slice_sender<T: RecorderSample>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<Arc<[T]>>,
) -> Result<AudioDevice<FanoutRecorder<UseArc<T>>>, RibbleWhisperError> {
    let audio_stream = audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| FanoutRecorder::new_arc(audio_sender),
        )
        .map_err(|e| {
            RibbleWhisperError::ParameterError(format!("Failed to build audio stream: {}", e))
        })?;

    Ok(audio_stream)
}
