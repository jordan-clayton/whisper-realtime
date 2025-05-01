use std::sync::Arc;
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::SyncSender;

use sdl2::audio::{AudioDevice, AudioFormatNum, AudioSpecDesired};
use sdl2::AudioSubsystem;

use crate::audio::recorder::{AudioRecorderSliceSender, AudioRecorderVecSender};
use crate::utils::errors::WhisperRealtimeError;

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
pub fn build_audio_stream<T: Default + Clone + Copy + Send + AudioFormatNum + 'static>(
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
pub fn build_audio_stream_using_slices<
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
pub fn build_audio_stream<T: Default + Clone + Copy + Send + AudioFormatNum + 'static>(
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
pub fn build_audio_stream_using_slice<
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
