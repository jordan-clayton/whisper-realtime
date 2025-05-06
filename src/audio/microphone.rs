use std::sync::Arc;
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::SyncSender;

use sdl2::{AudioSubsystem, Sdl};
use sdl2::audio::{AudioDevice, AudioFormatNum, AudioSpecDesired};

use crate::audio::recorder::{AudioRecorderSliceSender, AudioRecorderVecSender};
use crate::utils::errors::WhisperRealtimeError;

/// Basic Audio Backend that uses SDL to gain access to the microphone
/// At this time, there is no support for other audio backends, but this may happen in the future.
#[derive(Clone)]
pub struct AudioBackend {
    sdl_ctx: Arc<Sdl>,
    audio_subsystem: AudioSubsystem,
}

/// Request a microphone (feed) from the audio backend and build appropriately.
/// Call build_microphone_vec_sender if you happen to require sending/receiving audio chunks using
/// vectors. Otherwise, default to using shared slices.
impl AudioBackend {
    /// Initializes the audio backend to access the microphone when running a realtime transcription
    /// It is not encouraged to call new more than once, but it is not an error to do so.
    /// Returns an error if the backend fails to initialize.
    pub fn new() -> Result<Self, WhisperRealtimeError> {
        let ctx = sdl2::init();
        if let Err(reason) = &ctx {
            return Err(WhisperRealtimeError::ParameterError(format!(
                "Failed to create SDL context, error: {}",
                reason
            )));
        };
        let ctx = ctx.unwrap();
        let audio_subsystem = ctx.audio();
        if let Err(reason) = &audio_subsystem {
            return Err(WhisperRealtimeError::ParameterError(format!(
                "Failed to initialize audio subsystem, error: {}",
                reason
            )));
        };
        let audio_subsystem = audio_subsystem.unwrap();

        let sdl_ctx = Arc::new(ctx);
        Ok(Self {
            sdl_ctx,
            audio_subsystem,
        })
    }

    // If access to the sdl_ctx or the audio_subsystem are required, use the following to obtain a copy
    pub fn sdl_ctx(&self) -> Arc<Sdl> {
        self.sdl_ctx.clone()
    }
    pub fn audio_subsystem(&self) -> AudioSubsystem {
        self.audio_subsystem.clone()
    }

    #[cfg(feature = "crossbeam")]
    pub fn build_microphone<T: Default + Clone + Copy + Send + Sync + AudioFormatNum + 'static>(
        &self,
        audio_sender: crossbeam::channel::Sender<Arc<[T]>>,
    ) -> MicrophoneBuilder<Arc<[T]>> {
        MicrophoneBuilder::<Arc<[T]>>::new(&self.audio_subsystem, audio_sender)
    }

    #[cfg(not(feature = "crossbeam"))]
    pub fn build_microphone<T: Default + Clone + Copy + Send + Sync + AudioFormatNum + 'static>(
        &self,
        audio_sender: SyncSender<Arc<[T]>>,
    ) -> MicrophoneBuilder<Arc<[T]>> {
        MicrophoneBuilder::<Arc<[T]>>::new(&self.audio_subsystem, audio_sender)
    }
    #[cfg(feature = "crossbeam")]
    pub fn build_microphone_vec_sender<
        T: Default + Clone + Copy + Send + AudioFormatNum + 'static,
    >(
        &self,
        audio_sender: crossbeam::channel::Sender<Vec<T>>,
    ) -> MicrophoneBuilder<Vec<T>> {
        MicrophoneBuilder::<Vec<T>>::new(&self.audio_subsystem, audio_sender)
    }

    #[cfg(not(feature = "crossbeam"))]
    pub fn build_microphone_vec_sender<
        T: Default + Clone + Copy + Send + AudioFormatNum + 'static,
    >(
        &self,
        audio_sender: SyncSender<Vec<T>>,
    ) -> MicrophoneBuilder<Vec<T>> {
        MicrophoneBuilder::<Vec<T>>::new(&self.audio_subsystem, audio_sender)
    }
}

#[cfg(feature = "crossbeam")]
pub struct MicrophoneBuilder<'a, T> {
    audio_subsystem: &'a AudioSubsystem,
    audio_spec_desired: AudioSpecDesired,
    audio_sender: crossbeam::channel::Sender<T>,
}

#[cfg(not(feature = "crossbeam"))]
pub struct MicrophoneBuilder<'a, T> {
    audio_subsystem: &'a AudioSubsystem,
    audio_spec_desired: AudioSpecDesired,
    audio_sender: SyncSender<T>,
}

impl<'a, T> MicrophoneBuilder<'a, T> {
    pub fn with_audio_subsystem(mut self, audio_subsystem: &'a AudioSubsystem) -> Self {
        self.audio_subsystem = audio_subsystem;
        self
    }
    pub fn with_sample_rate(mut self, sample_rate: Option<i32>) -> Self {
        self.audio_spec_desired.freq = sample_rate;
        self
    }

    pub fn with_num_channels(mut self, num_channels: Option<u8>) -> Self {
        self.audio_spec_desired.channels = num_channels;
        self
    }
    /// Audio buffer size: must be a power of two.
    /// An invalid number of samples will default to the device sample size
    pub fn with_sample_size(mut self, samples: Option<u16>) -> Self {
        let sample_size = if let Some(size) = samples {
            if size.is_power_of_two() {
                Some(size)
            } else {
                None
            }
        } else {
            None
        };
        self.audio_spec_desired.samples = sample_size;
        self
    }

    pub fn with_desired_audio_spec(mut self, spec: AudioSpecDesired) -> Self {
        self.audio_spec_desired = spec;
        self
    }

    #[cfg(feature = "crossbeam")]
    pub fn with_vec_sender<S: Default + Clone + Copy + Send + AudioFormatNum + 'static>(
        self,
        sender: crossbeam::channel::Sender<Vec<S>>,
    ) -> MicrophoneBuilder<'a, Vec<S>> {
        MicrophoneBuilder {
            audio_subsystem: self.audio_subsystem,
            audio_spec_desired: self.audio_spec_desired,
            audio_sender: sender,
        }
    }

    #[cfg(not(feature = "crossbeam"))]
    pub fn with_vec_sender<S: Default + Clone + Copy + Send + AudioFormatNum + 'static>(
        self,
        sender: SyncSender<Vec<S>>,
    ) -> MicrophoneBuilder<'a, Vec<S>> {
        MicrophoneBuilder {
            audio_subsystem: self.audio_subsystem,
            audio_spec_desired: self.audio_spec_desired,
            audio_sender: sender,
        }
    }

    #[cfg(feature = "crossbeam")]
    pub fn with_slice_sender<S: Default + Clone + Copy + Send + Sync + AudioFormatNum + 'static>(
        self,
        sender: crossbeam::channel::Sender<Arc<[S]>>,
    ) -> MicrophoneBuilder<'a, Arc<[S]>> {
        MicrophoneBuilder {
            audio_subsystem: self.audio_subsystem,
            audio_spec_desired: self.audio_spec_desired,
            audio_sender: sender,
        }
    }

    #[cfg(not(feature = "crossbeam"))]
    pub fn with_slice_sender<S: Default + Clone + Copy + Send + Sync + AudioFormatNum + 'static>(
        self,
        sender: SyncSender<Arc<[S]>>,
    ) -> MicrophoneBuilder<'a, Arc<[S]>> {
        MicrophoneBuilder {
            audio_subsystem: self.audio_subsystem,
            audio_spec_desired: self.audio_spec_desired,
            audio_sender: sender,
        }
    }
}

impl<'a, T: Default + Clone + Copy + Send + AudioFormatNum + 'static>
    MicrophoneBuilder<'a, Vec<T>>
{
    #[cfg(feature = "crossbeam")]
    pub fn new(
        audio_subsystem: &'a AudioSubsystem,
        audio_sender: crossbeam::channel::Sender<Vec<T>>,
    ) -> Self {
        let audio_spec_desired = AudioSpecDesired {
            freq: None,
            channels: None,
            samples: None,
        };
        Self {
            audio_subsystem,
            audio_spec_desired,
            audio_sender,
        }
    }

    #[cfg(not(feature = "crossbeam"))]
    pub fn new(audio_subsystem: &'a AudioSubsystem, audio_sender: SyncSender<Vec<T>>) -> Self {
        let audio_spec_desired = AudioSpecDesired {
            freq: None,
            channels: None,
            samples: None,
        };
        Self {
            audio_subsystem,
            audio_spec_desired,
            audio_sender,
        }
    }

    pub fn build(self) -> Result<AudioDevice<AudioRecorderVecSender<T>>, WhisperRealtimeError> {
        build_audio_stream_vec_sender(
            &self.audio_subsystem,
            &self.audio_spec_desired,
            self.audio_sender,
        )
    }
}

impl<'a, T: Default + Clone + Copy + Send + Sync + AudioFormatNum + 'static>
    MicrophoneBuilder<'a, Arc<[T]>>
{
    #[cfg(feature = "crossbeam")]
    pub fn new(
        audio_subsystem: &'a AudioSubsystem,
        audio_sender: crossbeam::channel::Sender<Arc<[T]>>,
    ) -> Self {
        let audio_spec_desired = AudioSpecDesired {
            freq: None,
            channels: None,
            samples: None,
        };
        Self {
            audio_subsystem,
            audio_spec_desired,
            audio_sender,
        }
    }
    #[cfg(not(feature = "crossbeam"))]
    pub fn new(audio_subsystem: &'a AudioSubsystem, audio_sender: SyncSender<Arc<[T]>>) -> Self {
        let audio_spec_desired = AudioSpecDesired {
            freq: None,
            channels: None,
            samples: None,
        };
        Self {
            audio_subsystem,
            audio_spec_desired,
            audio_sender,
        }
    }
    pub fn build(self) -> Result<AudioDevice<AudioRecorderSliceSender<T>>, WhisperRealtimeError> {
        build_audio_stream_slice_sender(
            &self.audio_subsystem,
            &self.audio_spec_desired,
            self.audio_sender,
        )
    }
}

/// The following functions are exposed but their use is not encouraged unless required.
/// Prefer the AudioBackend and MicrophoneBuilder API wherever possible.
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

#[cfg(not(feature = "crossbeam"))]
#[inline]
pub fn build_audio_stream_vec_sender<
    T: Default + Clone + Copy + Send + AudioFormatNum + 'static,
>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: SyncSender<Vec<T>>,
) -> Result<AudioDevice<AudioRecorderVecSender<T>>, WhisperRealtimeError> {
    let audio_stream = audio_subsystem.open_capture(
        // Device - should be default, SDL should change if the user changes devices in their sysprefs.
        None,
        desired_spec,
        |_spec| AudioRecorderVecSender {
            sender: audio_sender,
        },
    );
    match audio_stream {
        Err(e) => Err(WhisperRealtimeError::ParameterError(format!(
            "Failed to build audio stream: {}",
            e
        ))),
        Ok(audio) => Ok(audio),
    }
}

#[cfg(not(feature = "crossbeam"))]
#[inline]
pub fn build_audio_stream_slice_sender<
    T: Default + Clone + Copy + Send + Sync + AudioFormatNum + 'static,
>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: SyncSender<Arc<[T]>>,
) -> Result<AudioDevice<AudioRecorderSliceSender<T>>, WhisperRealtimeError> {
    let audio_stream = audio_subsystem.open_capture(
        // Device - should be default, SDL should change if the user changes devices in their sysprefs.
        None,
        desired_spec,
        |_spec| AudioRecorderSliceSender {
            sender: audio_sender,
        },
    );
    match audio_stream {
        Err(e) => Err(WhisperRealtimeError::ParameterError(format!(
            "Failed to build audio stream: {}",
            e
        ))),
        Ok(audio) => Ok(audio),
    }
}

#[cfg(feature = "crossbeam")]
#[inline]
pub fn build_audio_stream_vec_sender<
    T: Default + Clone + Copy + Send + AudioFormatNum + 'static,
>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: crossbeam::channel::Sender<Vec<T>>,
) -> Result<AudioDevice<AudioRecorderVecSender<T>>, WhisperRealtimeError> {
    let audio_stream = audio_subsystem.open_capture(
        // Device - should be default, SDL should change if the user changes devices in their sysprefs.
        None,
        desired_spec,
        |_spec| AudioRecorderVecSender {
            sender: audio_sender,
        },
    );
    match audio_stream {
        Err(e) => Err(WhisperRealtimeError::ParameterError(format!(
            "Failed to build audio stream: {}",
            e
        ))),
        Ok(audio) => Ok(audio),
    }
}

#[cfg(feature = "crossbeam")]
#[inline]
pub fn build_audio_stream_slice_sender<
    T: Default + Clone + Copy + Send + Sync + AudioFormatNum + 'static,
>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: crossbeam::channel::Sender<Arc<[T]>>,
) -> Result<AudioDevice<AudioRecorderSliceSender<T>>, WhisperRealtimeError> {
    let audio_stream = audio_subsystem.open_capture(
        // Device - should be default, SDL should change if the user changes devices in their sysprefs.
        None,
        desired_spec,
        |_spec| AudioRecorderSliceSender {
            sender: audio_sender,
        },
    );
    match audio_stream {
        Err(e) => Err(WhisperRealtimeError::ParameterError(format!(
            "Failed to build audio stream: {}",
            e
        ))),
        Ok(audio) => Ok(audio),
    }
}
