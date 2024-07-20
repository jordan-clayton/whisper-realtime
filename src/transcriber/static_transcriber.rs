use std::error::Error;
use std::ffi::c_int;
use std::sync::{Arc, Mutex};

use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::configs::Configs;
use crate::errors::TranscriptionError;
use crate::traits::Transcriber;

// TODO: Determine whether to use generics or require f32.
pub struct StaticTranscriber {
    configs: Arc<Configs>,
    audio: Arc<Mutex<Vec<f32>>>,
}

impl StaticTranscriber {
    pub fn new(audio: Arc<Mutex<Vec<f32>>>) -> Self {
        StaticTranscriber {
            configs: Arc::new(Configs::default()),
            audio,
        }
    }

    pub fn new_with_configs(audio: Arc<Mutex<Vec<f32>>>, configs: Arc<Configs>) -> Self {
        StaticTranscriber { configs, audio }
    }

    fn send_error_string(e: &impl Error) -> String {
        let mut error_string = String::from("Error");
        error_string.push_str(&e.to_string());
        return error_string;
    }
}

impl Transcriber for StaticTranscriber {
    fn process_audio(&mut self, whisper_state: &mut WhisperState) -> String {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        Self::set_full_params(&mut params, &self.configs, None);

        let audio_samples = self.audio.lock().expect("failed to grab mutex");

        let mono_audio_samples = whisper_rs::convert_stereo_to_mono_audio(&audio_samples)
            .expect("failed to convert to mono");

        let result = whisper_state.full(params, &mono_audio_samples);

        if let Err(e) = result {
            return Self::send_error_string(&e);
        };

        let num_segments = whisper_state
            .full_n_segments()
            .expect("failed to get segments");

        if num_segments == 0 {
            let error =
                TranscriptionError::new_with_reason(String::from("Zero segments transcribed"));

            return Self::send_error_string(&error);
        };

        let mut text: Vec<String> = vec![];
        for i in 0..num_segments {
            let segment = whisper_state
                .full_get_segment_text(i)
                .expect("failed to get segment");

            text.push(segment);
        }

        let text = text.join("");
        let text = text.trim();
        let text_string = String::from(text);

        text_string.clone()
    }

    fn set_full_params<'a>(
        full_params: &mut FullParams<'a, 'a>,
        prefs: &'a Configs,
        tokens: Option<&'a Vec<c_int>>,
    ) {
        let lang = prefs.set_language.as_ref();

        full_params.set_n_threads(prefs.n_threads);
        full_params.set_n_threads(prefs.n_threads);
        full_params.set_translate(prefs.set_translate);

        if lang.is_some() {
            full_params.set_language(Some(lang.unwrap().as_str()));
        } else {
            full_params.set_language(Some("auto"))
        }

        // // Stdio only
        full_params.set_print_special(prefs.print_special);
        full_params.set_print_progress(prefs.print_progress);
        full_params.set_print_realtime(prefs.print_realtime);
        full_params.set_print_timestamps(prefs.print_timestamps);

        if tokens.is_some() {
            let token_buffer = tokens.unwrap();
            full_params.set_tokens(&token_buffer)
        }
    }
}
