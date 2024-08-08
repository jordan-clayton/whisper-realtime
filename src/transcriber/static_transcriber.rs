use std::error::Error;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc;

use lazy_static::lazy_static;
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::configs::Configs;
use crate::errors::{WhisperRealtimeError, WhisperRealtimeErrorType};

use super::transcriber::Transcriber;

// Workaround for whisper-rs issue #134 -- moving memory in rust causes a segmentation fault
// in the progress callback.
// Solution from: https://github.com/thewh1teagle/vibe/blob/main/core/src/transcribe.rs

lazy_static! {
    static ref PROGRESS_CALLBACK: Mutex<Option<Box<dyn FnMut(i32) + Send + Sync>>> =
        Mutex::new(None);
}

pub enum SupportedAudioSample {
    I16(Vec<i16>),
    F32(Vec<f32>),
}

#[derive(Copy, Clone, PartialEq)]
pub enum SupportedChannels {
    MONO,
    STEREO,
}

#[cfg(not(feature = "crossbeam"))]
pub struct StaticTranscriber {
    configs: Arc<Configs>,
    data_sender: Option<mpsc::Sender<Result<(String, bool), WhisperRealtimeError>>>,
    audio: Arc<Mutex<SupportedAudioSample>>,
    channels: SupportedChannels,
}

#[cfg(feature = "crossbeam")]
pub struct StaticTranscriber {
    configs: Arc<Configs>,
    data_sender: Option<crossbeam::channel::Sender<Result<(String, bool), WhisperRealtimeError>>>,
    audio: Arc<Mutex<SupportedAudioSample>>,
    channels: SupportedChannels,
}

impl StaticTranscriber {
    #[cfg(not(feature = "crossbeam"))]
    pub fn new(
        audio: Arc<Mutex<SupportedAudioSample>>,
        data_sender: Option<mpsc::Sender<Result<(String, bool), WhisperRealtimeError>>>,
        channels: SupportedChannels,
    ) -> Self {
        Self {
            configs: Arc::<Configs>::default(),
            data_sender,
            audio,
            channels,
        }
    }

    #[cfg(feature = "crossbeam")]
    pub fn new(
        audio: Arc<Mutex<SupportedAudioSample>>,
        data_sender: Option<
            crossbeam::channel::Sender<Result<(String, bool), WhisperRealtimeError>>,
        >,
        channels: SupportedChannels,
    ) -> Self {
        Self {
            configs: Arc::<Configs>::default(),
            data_sender,
            audio,
            channels,
        }
    }

    #[cfg(not(feature = "crossbeam"))]
    pub fn new_with_configs(
        audio: Arc<Mutex<SupportedAudioSample>>,
        data_sender: Option<mpsc::Sender<Result<(String, bool), WhisperRealtimeError>>>,
        configs: Arc<Configs>,
        channels: SupportedChannels,
    ) -> Self {
        Self {
            configs,
            data_sender,
            audio,
            channels,
        }
    }

    #[cfg(feature = "crossbeam")]
    pub fn new_with_configs(
        audio: Arc<Mutex<SupportedAudioSample>>,
        data_sender: Option<
            crossbeam::channel::Sender<Result<(String, bool), WhisperRealtimeError>>,
        >,
        configs: Arc<Configs>,
        channels: SupportedChannels,
    ) -> Self {
        Self {
            configs,
            data_sender,
            audio,
            channels,
        }
    }
    fn send_error_string(e: &impl Error) -> String {
        let mut error_string = String::from("Error");
        error_string.push_str(&e.to_string());
        return error_string;
    }
}

impl Transcriber for StaticTranscriber {
    fn process_audio(
        &mut self,
        whisper_state: &mut WhisperState,
        progress_callback: Option<impl FnMut(i32) + Send + Sync + 'static>,
    ) -> String {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        Self::set_full_params(&mut params, &self.configs, None);

        if let Some(callback) = progress_callback {
            {
                let mut guard = PROGRESS_CALLBACK
                    .lock()
                    .expect("Failed to get callback lock");

                *guard = Some(Box::new(callback));
            }

            params.set_progress_callback_safe(|p| {
                let mut progress_callback = PROGRESS_CALLBACK
                    .lock()
                    .expect("Failed to get callback mutex");

                if let Some(callback) = progress_callback.as_mut() {
                    callback(p)
                }
            })
        }

        let mut guard = self.audio.lock().expect("Failed to grab mutex");

        let audio_samples = guard.deref_mut();

        let audio_samples = match audio_samples {
            SupportedAudioSample::I16(int_samples) => {
                let len = int_samples.len();
                let mut float_samples = vec![0.0f32; len];
                whisper_rs::convert_integer_to_float_audio(int_samples, &mut float_samples)
                    .expect("Failed to convert audio from int to float");
                float_samples
            }
            SupportedAudioSample::F32(float_samples) => float_samples.clone(),
        };

        let mono_audio_samples = match self.channels {
            SupportedChannels::MONO => audio_samples,
            SupportedChannels::STEREO => whisper_rs::convert_stereo_to_mono_audio(&audio_samples)
                .expect("failed to convert stereo audio to mono"),
        };

        let result = whisper_state.full(params, &mono_audio_samples);

        if let Err(e) = result {
            if let Some(ref data_sender) = self.data_sender {
                let err = WhisperRealtimeError::new(
                    WhisperRealtimeErrorType::TranscriptionError,
                    format!("Model failure. Error: {}", e),
                );

                data_sender.send(Err(err)).expect("Data channel closed")
            }

            return Self::send_error_string(&e);
        };

        let num_segments = whisper_state
            .full_n_segments()
            .expect("failed to get segments");

        if num_segments == 0 {
            let error = WhisperRealtimeError::new(
                WhisperRealtimeErrorType::TranscriptionError,
                String::from("Zero segments transcribed"),
            );

            return Self::send_error_string(&error);
        };

        let mut text: Vec<String> = vec![];
        for i in 0..num_segments {
            let segment = whisper_state
                .full_get_segment_text(i)
                .expect("failed to get segment");

            if let Some(ref data_sender) = self.data_sender {
                data_sender
                    .send(Ok((segment.clone(), true)))
                    .expect("Data channel closed")
            }

            text.push(segment);
        }

        // Static segments are generally "longer" and thus are separated by newline.
        let text = text.join("\n");
        let text = text.trim();
        let text_string = String::from(text);

        text_string.clone()
    }
}
