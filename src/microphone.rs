#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::SyncSender;

use sdl2::{
    audio::{AudioDevice, AudioFormatNum, AudioSpecDesired},
    AudioSubsystem,
};

use crate::recorder::Recorder;

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
pub fn build_audio_stream<T: Default + Clone + Copy + Send + AudioFormatNum + 'static>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: SyncSender<Vec<T>>,
) -> AudioDevice<Recorder<T>> {
    audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| Recorder {
                sender: audio_sender,
            },
        )
        .expect("failed to build audio stream")
}

#[cfg(feature = "crossbeam")]
pub fn build_audio_stream<T: Default + Clone + Copy + Send + AudioFormatNum + 'static>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    audio_sender: crossbeam::channel::Sender<Vec<T>>,
) -> AudioDevice<Recorder<T>> {
    audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| Recorder {
                sender: audio_sender,
            },
        )
        .expect("failed to build audio stream")
}
