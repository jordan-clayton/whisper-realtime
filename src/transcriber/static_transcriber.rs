use std::error::Error;
use std::ffi::c_void;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc;
use std::sync::Mutex;

use lazy_static::lazy_static;
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::transcriber::traits::Transcriber;
use crate::utils::errors::WhisperRealtimeError;
use crate::whisper::configs::Configs;

// Workaround for whisper-rs issue #134 -- moving memory in rust causes a segmentation fault
// in the progress callback.
// Solution from: https://github.com/thewh1teagle/vibe/blob/main/core/src/transcribe.rs

lazy_static! {
    static ref PROGRESS_CALLBACK: Mutex<Option<Box<dyn FnMut(i32) + Send + Sync>>> =
        Mutex::new(None);
}

// TODO: these probably shouldn't be vectors; use [T]
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

// TODO: get rid of the mutex
// TODO: rename to OfflineTranscriber
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
        let mut error_string = String::from("Error:");
        error_string.push_str(&e.to_string());
        return error_string;
    }
}

impl Transcriber for StaticTranscriber {
    fn process_audio(
        &mut self,
        whisper_state: &mut WhisperState,
        run_transcription: Arc<AtomicBool>,
        progress_callback: Option<impl FnMut(i32) + Send + Sync + 'static>,
    ) -> String {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        Self::set_full_params(&mut params, &self.configs, None);

        // At the moment, the
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

        // Set the trigger for aborting whisper transcription
        let start_encoder_callback_user_data = run_transcription.as_ptr() as *mut c_void;

        // These are unsafe insofar that they touch C
        // The underlying C code is safe provided the callback doesn't touch the whisper pointer.
        unsafe {
            params.set_start_encoder_callback(Some(check_running_state));
            params.set_start_encoder_callback_user_data(start_encoder_callback_user_data);
        }

        let mut guard = self.audio.lock().expect("Poisoned mutex");

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
                let err = WhisperRealtimeError::TranscriptionError(format!("Whisper Error: {}", e));

                data_sender.send(Err(err)).expect("Data channel closed")
            }

            return Self::send_error_string(&e);
        };

        let num_segments = whisper_state.full_n_segments().unwrap_or(0);

        if num_segments == 0 {
            let error =
                WhisperRealtimeError::TranscriptionError("Zero segments transcribed".to_owned());

            return Self::send_error_string(&error);
        };

        let mut text: Vec<String> = vec![];
        for i in 0..num_segments {
            if let Ok(segment) = whisper_state.full_get_segment_text(i) {
                if let Some(ref data_sender) = self.data_sender {
                    data_sender
                        .send(Ok((segment.clone(), true)))
                        .expect("Data channel closed")
                }

                text.push(segment);
            }
        }

        // Static segments are generally "longer" and thus are separated by newline.
        let text = text.join("\n");
        let text = text.trim();
        let text_string = String::from(text);

        text_string.clone()
    }
}

extern "C" fn check_running_state(
    _: *mut whisper_rs_sys::whisper_context,
    _: *mut whisper_rs_sys::whisper_state,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let run_transcription = AtomicBool::from_ptr(user_data as *mut bool);
        run_transcription.load(Ordering::Acquire)
    }
}
