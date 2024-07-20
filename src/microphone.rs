use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::SyncSender;

use sdl2::audio::{AudioDevice, AudioFormatNum, AudioSpecDesired};
use sdl2::AudioSubsystem;

use crate::recorder::Recorder;

// AudioSpecDesired {
//     freq: Option<i32>, WHISPER_SAMPLE_RATE as i32,
//     channels: Option<u8>, 1
//     samples: Option<u16>, 1024
// }

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

pub fn build_audio_stream<T: Default + Clone + Copy + Send + AudioFormatNum + 'static>(
    audio_subsystem: &AudioSubsystem,
    desired_spec: &AudioSpecDesired,
    text_sender: SyncSender<Vec<T>>,
    is_running: Arc<AtomicBool>,
) -> AudioDevice<Recorder<T>> {
    audio_subsystem
        .open_capture(
            // Device - should be default, SDL should change if the user changes devices in their sysprefs.
            None,
            desired_spec,
            |_spec| Recorder {
                sender: text_sender,
                is_running,
            },
        )
        .expect("failed to build audio stream")
}
