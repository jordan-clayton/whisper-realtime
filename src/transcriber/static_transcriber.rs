use std::error::Error;
use std::sync::{Arc, Mutex};

use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::configs::Configs;
use crate::errors::{WhisperRealtimeError, WhisperRealtimeErrorType};
use super::transcriber::Transcriber;

// NOTE: At this time only f32 is supported. i16 may be implemented at a later time.
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
                WhisperRealtimeError::new(WhisperRealtimeErrorType::TranscriptionError, String::from("Zero segments transcribed"));

            return Self::send_error_string(&error);
        };

        let mut text: Vec<String> = vec![];
        for i in 0..num_segments {
            let segment = whisper_state
                .full_get_segment_text(i)
                .expect("failed to get segment");

            text.push(segment);
        }

        // Static segments are generally "longer" and thus are separated by newline.
        let text = text.join("\n");
        let text = text.trim();
        let text_string = String::from(text);

        text_string.clone()
    }
}
