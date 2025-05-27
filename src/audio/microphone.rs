use std::marker::PhantomData;
use std::sync::Arc;

use sdl2::{AudioSubsystem, Sdl};
use sdl2::audio::{AudioDevice, AudioSpecDesired};

use crate::audio::recorder::{AudioInputAdapter, AudioRecorder, RecorderSample, UseArc, UseVec};
use crate::utils::{constants, Sender};
use crate::utils::errors::WhisperRealtimeError;

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
    pub fn new() -> Result<Self, WhisperRealtimeError> {
        let ctx = sdl2::init().map_err(|e| {
            WhisperRealtimeError::ParameterError(format!(
                "Failed to create SDL context, error: {}",
                e
            ))
        })?;

        let audio_subsystem = ctx.audio().map_err(|e| {
            WhisperRealtimeError::ParameterError(format!(
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

    /// To access the inner [sdl2::sdl] context
    pub fn sdl_ctx(&self) -> Arc<Sdl> {
        self.sdl_ctx.clone()
    }
    /// To access the inner [sdl2::sdl::AudioSubsystem]
    pub fn audio_subsystem(&self) -> AudioSubsystem {
        self.audio_subsystem.clone()
    }

    /// A convenience method that prepares [sdl2::audio::AudioDevice] for use
    /// with [crate::transcriber::realtime_transcriber::RealtimeTranscriber] to transcribe
    /// audio realtime.
    /// This sends audio out as Arc<[T]> because it's the most efficient. Use a builder if
    /// vectors are required. See: [crate::audio::microphone::AudioBackend::build_microphone_vec]
    /// # Arguments:
    /// * audio_sender: a message sender to forward audio from the input device
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err(WhisperRealtimeError) on failure to build.
    /// See: [crate::audio::microphone::MicrophoneBuilder] for error conditions.
    pub fn build_whisper_default<T: RecorderSample>(
        &self,
        audio_sender: Sender<Arc<[T]>>,
    ) -> Result<Microphone<T, UseArc>, WhisperRealtimeError> {
        self.build_microphone(audio_sender)
            .with_num_channels(Some(1))
            .with_sample_rate(Some(constants::WHISPER_SAMPLE_RATE as i32))
            .with_sample_size(Some(constants::AUDIO_BUFFER_SIZE))
            .build()
    }

    /// The "default" way to start building an audio capture
    /// Returns a builder to set up the audio capture parameters with a callback that sends audio
    /// using Arc<[T]>
    /// Prefer this over Vec<T> unless Vectors are absolutely required.
    pub fn build_microphone<T: RecorderSample>(
        &self,
        audio_sender: Sender<Arc<[T]>>,
    ) -> MicrophoneBuilder<T, UseArc> {
        self.build_microphone_arc(audio_sender)
    }

    /// Returns a builder to set up the audio capture parameters with a callback that sends audio
    /// using Arc<[T]>
    pub fn build_microphone_arc<T: RecorderSample>(
        &self,
        audio_sender: Sender<Arc<[T]>>,
    ) -> MicrophoneBuilder<T, UseArc> {
        MicrophoneBuilder::new_arc(&self.audio_subsystem, audio_sender)
    }

    /// Returns a builder to set up the audio capture parameters with a callback that sends audio
    /// using Vec<[T]>
    pub fn build_microphone_vec<T: RecorderSample>(
        &self,
        audio_sender: Sender<Vec<T>>,
    ) -> MicrophoneBuilder<T, UseVec> {
        MicrophoneBuilder::new_vec(&self.audio_subsystem, audio_sender)
    }
}

/// A builder for setting audio input configurations
#[derive(Clone)]
pub struct MicrophoneBuilder<'a, T, AC>
where
    T: RecorderSample,
    AC: AudioInputAdapter<T> + Send + Clone,
{
    audio_subsystem: &'a AudioSubsystem,
    audio_spec_desired: AudioSpecDesired,
    /// Used in the [sdl2::audio::AudioCallback] to forward input audio
    audio_sender: Sender<AC::SenderOutput>,
    _marker: PhantomData<AC>,
}

impl<'a, T, AC> MicrophoneBuilder<'a, T, AC>
where
    T: RecorderSample,
    AC: AudioInputAdapter<T> + Clone + Send,
{
    pub fn new(
        audio_subsystem: &'a AudioSubsystem,
        audio_sender: Sender<AC::SenderOutput>,
    ) -> Self {
        // AudioSpecDesired does not implement Default
        let audio_spec_desired = AudioSpecDesired {
            freq: None,
            channels: None,
            samples: None,
        };

        Self {
            audio_subsystem,
            audio_spec_desired,
            audio_sender,
            _marker: Default::default(),
        }
    }
    /// To change the [sdl2::sdl::AudioSubsystem]
    pub fn with_audio_subsystem(mut self, audio_subsystem: &'a AudioSubsystem) -> Self {
        self.audio_subsystem = audio_subsystem;
        self
    }

    /// To change the desired sample rate
    pub fn with_sample_rate(mut self, sample_rate: Option<i32>) -> Self {
        self.audio_spec_desired.freq = sample_rate;
        self
    }

    /// To change the desired number of channels (eg. 1 = mono, 2 = stereo)
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
    pub fn with_desired_audio_spec(mut self, spec: AudioSpecDesired) -> Self {
        self.audio_spec_desired = spec;
        self
    }

    /// To set the audio callback sender.
    pub fn with_vec_sender<S: RecorderSample>(
        self,
        sender: Sender<Vec<S>>,
    ) -> MicrophoneBuilder<'a, S, UseVec> {
        MicrophoneBuilder {
            audio_subsystem: self.audio_subsystem,
            audio_spec_desired: self.audio_spec_desired,
            audio_sender: sender,
            _marker: Default::default(),
        }
    }

    /// To set the audio callback sender.
    pub fn with_arc_sender<S: RecorderSample>(
        self,
        sender: Sender<Arc<[S]>>,
    ) -> MicrophoneBuilder<'a, S, UseArc> {
        MicrophoneBuilder {
            audio_subsystem: self.audio_subsystem,
            audio_spec_desired: self.audio_spec_desired,
            audio_sender: sender,
            _marker: Default::default(),
        }
    }

    /// Builds [sdl2::audio::AudioDevice] to open audio capture (eg. for use in realtime transcription)
    /// # Returns:
    /// * Ok(AudioDevice) on success, Err on SDL failure.
    pub fn build(self) -> Result<Microphone<T, AC>, WhisperRealtimeError> {
        let device = build_audio_stream(
            &self.audio_subsystem,
            &self.audio_spec_desired,
            self.audio_sender,
        )?;
        Ok(Microphone { device })
    }
}

impl<'a, T: RecorderSample> MicrophoneBuilder<'a, T, UseVec> {
    pub fn new_vec(audio_subsystem: &'a AudioSubsystem, audio_sender: Sender<Vec<T>>) -> Self {
        Self::new(audio_subsystem, audio_sender)
    }
}

impl<'a, T: RecorderSample> MicrophoneBuilder<'a, T, UseArc> {
    pub fn new_arc(audio_subsystem: &'a AudioSubsystem, audio_sender: Sender<Arc<[T]>>) -> Self {
        Self::new(audio_subsystem, audio_sender)
    }
}

/// Encapsulates [sdl2::audio::AudioDevice]. This may become a trait if/when multiple audio backends are needed.
/// NOTE: this does not implement Sync or Send; construct this on the thread that will be opening
/// and closing the audio capture.
pub struct Microphone<T, AC>
where
    T: RecorderSample,
    AC: AudioInputAdapter<T> + Clone + Send,
{
    device: AudioDevice<AudioRecorder<T, AC>>,
}

impl<T, AC> Microphone<T, AC>
where
    T: RecorderSample,
    AC: AudioInputAdapter<T> + Clone + Send,
{
    pub fn play(&self) {
        self.device.resume()
    }
    pub fn pause(&self) {
        self.device.pause()
    }
}

// The following functions are exposed but their use is not encouraged unless required.

/// This is deprecated and will be removed at a later date.
/// Prefer [crate::audio::microphone::MicrophoneBuilder].
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
/// Prefer [crate::audio::microphone::MicrophoneBuilder]
#[inline]
pub fn build_audio_stream<T: RecorderSample, AC: AudioInputAdapter<T> + Send>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<AC::SenderOutput>,
) -> Result<AudioDevice<AudioRecorder<T, AC>>, WhisperRealtimeError> {
    let audio_stream = audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| AudioRecorder::new(audio_sender),
        )
        .map_err(|e| {
            WhisperRealtimeError::ParameterError(format!("Failed to build audio stream: {}", e))
        })?;

    Ok(audio_stream)
}

/// Visibility is currently exposed due to legacy implementations.
/// Do not rely on this function, as it will be marked private in the future.
/// Prefer [crate::audio::microphone::MicrophoneBuilder]
#[inline]
pub fn build_audio_stream_vec_sender<T: RecorderSample>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<Vec<T>>,
) -> Result<AudioDevice<AudioRecorder<T, UseVec>>, WhisperRealtimeError> {
    let audio_stream = audio_subsystem.open_capture(
        // Device - should be default, SDL should change if the user changes devices in their sysprefs.
        None,
        desired_spec,
        |_spec| AudioRecorder::new_vec(audio_sender),
    );
    match audio_stream {
        Err(e) => Err(WhisperRealtimeError::ParameterError(format!(
            "Failed to build audio stream: {}",
            e
        ))),
        Ok(audio) => Ok(audio),
    }
}

/// Visibility is currently exposed due to legacy implementations.
/// Do not rely on this function, as it will be marked private in the future.
/// Prefer [crate::audio::microphone::MicrophoneBuilder]
#[inline]
pub fn build_audio_stream_slice_sender<T: RecorderSample>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: Sender<Arc<[T]>>,
) -> Result<AudioDevice<AudioRecorder<T, UseArc>>, WhisperRealtimeError> {
    let audio_stream = audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| AudioRecorder::new_arc(audio_sender),
        )
        .map_err(|e| {
            WhisperRealtimeError::ParameterError(format!("Failed to build audio stream: {}", e))
        })?;

    Ok(audio_stream)
}
