#![allow(clippy::uninlined_format_args)]

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::thread::sleep;
use std::time::Duration;

use voice_activity_detector::VoiceActivityDetector;
use whisper_rs::{FullParams, SamplingStrategy, WhisperError, WhisperState};

use crate::audio_ring_buffer::AudioRingBuffer;
use crate::configs::Configs;
use crate::constants;
use crate::errors::{WhisperRealtimeError, WhisperRealtimeErrorType};
use super::transcriber::Transcriber;

/// This implementation is a modified port of the whisper.cpp stream example, see: 
/// https://github.com/ggerganov/whisper.cpp/blob/master/examples/stream/stream.cpp
///
/// Realtime on CPU has not yet been tested and may or may not be feasible. 
/// Building with GPU support is currently recommended.

// TODO: refactor the hardcoded constants into RealtimeConfigs
pub struct RealtimeTranscriber {
    configs: Arc<Configs>,
    audio: Arc<AudioRingBuffer<f32>>,
    // This might be wise to factor out.
    output_buffer: Vec<String>,

    // To send data to the G/UI
    data_sender: Arc<Sender<Result<(String, bool), WhisperRealtimeError>>>,
    // TODO: re-implement use of token buffer once optional VAD has been reimplemented.
    token_buffer: Vec<std::ffi::c_int>,
    ready: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    vad: VoiceActivityDetector,
}

impl RealtimeTranscriber {
    pub fn new(
        audio: Arc<AudioRingBuffer<f32>>,
        data_sender: Arc<Sender<Result<(String, bool), WhisperRealtimeError>>>,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
    ) -> Self {
        let output_buffer: Vec<String> = vec![];
        let token_buffer: Vec<std::ffi::c_int> = vec![];
        let vad = VoiceActivityDetector::builder()
            .sample_rate(constants::WHISPER_SAMPLE_RATE as i64)
            .chunk_size(1024usize)
            .build()
            .expect("failed to build voice activity detector");

        RealtimeTranscriber {
            configs: Arc::new(Configs::default()),
            audio,
            output_buffer,
            data_sender,
            token_buffer,
            ready,
            running,
            vad,
        }
    }

    pub fn new_with_configs(
        audio: Arc<AudioRingBuffer<f32>>,
        data_sender: Arc<Sender<Result<(String, bool), WhisperRealtimeError>>>,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
        configs: Arc<Configs>,
    ) -> Self {
        let output_buffer: Vec<String> = vec![];
        let token_buffer: Vec<std::ffi::c_int> = vec![];
        let vad = VoiceActivityDetector::builder()
            .sample_rate(constants::WHISPER_SAMPLE_RATE as i64)
            .chunk_size(1024usize)
            .build()
            .expect("failed to build voice activity detector");
        RealtimeTranscriber {
            configs,
            audio,
            output_buffer,
            data_sender,
            token_buffer,
            ready,
            running,
            vad,
        }
    }

    // Ideally this should be in f32
    fn is_voice_detected<T: voice_activity_detector::Sample>(
        vad: &mut VoiceActivityDetector,
        audio_data: &Vec<T>,
        voice_threshold: f32,
        // last_ms: usize,
    ) -> bool {
        let samples = audio_data.clone();

        let probability = vad.predict(samples);

        return probability > voice_threshold;
    }
}

// TODO: re-implement non-vad use & parameterize VAD use.
impl Transcriber for RealtimeTranscriber {
    // Ideally this should be run on its own thread.
    fn process_audio(&mut self, whisper_state: &mut WhisperState) -> String {
        let ready = self.ready.clone().load(Ordering::Relaxed);
        if !ready {
            return String::from("");
        }

        let mut t_last = std::time::Instant::now();

        let mut audio_samples: Vec<f32> = vec![0f32; constants::N_SAMPLES_30S];
        let mut audio_samples_vad: Vec<f32> = vec![0f32; constants::N_SAMPLES_30S];

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        Self::set_full_params(&mut params, &self.configs, None);

        // TODO: Refactor to either set a flag or send a message.
        println!("Start Speaking: ");

        let mut pause_detected = true;
        let mut phrase_finished = false;

        let mut total_time: u128 = 0;

        loop {
            let running = self.running.clone().load(Ordering::Relaxed);
            if !running {

                self.data_sender
                    .send(Ok((String::from("\nEnd Transcription\n"), true)))
                    .expect("Failed to send transcription");
                break;
            }

            // Get the time & check for pauses -> phrase-complete?
            let t_now = std::time::Instant::now();

            let diff = t_now - t_last;
            let millis = diff.as_millis();
            total_time += millis;

            if millis < self.configs.vad_sample_ms as u128 {
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                continue;
            }

            // Perhaps this might be better if it were lower.
            if millis >= self.configs.phrase_timeout as u128 {
                phrase_finished = true;
            }

            self.audio
                .get_audio(self.configs.vad_sample_ms, &mut audio_samples_vad);

            // Vad is mono only.
            let new_samples = whisper_rs::convert_stereo_to_mono_audio(&audio_samples_vad)
                .expect("failed to convert new samples to mono");

            if Self::is_voice_detected(
                &mut self.vad,
                &new_samples,
                self.configs.voice_threshold,
                // 1000
            ) {
                self.audio
                    .get_audio(self.configs.audio_sample_ms, &mut audio_samples);

                pause_detected = false;
            } else {
                if !pause_detected {
                    pause_detected = true;
                    phrase_finished = true;
                    self.output_buffer.push(String::from("\n"));
                    self.data_sender
                        .send(Ok((String::from("\n"), true)))
                        .expect("Failed to send transcription");

                    // Clear the previous 10s of the audio (keep a small amount to seed the model).
                    // self.audio.clear_n_samples(constants::AUDIO_CHUNK_SIZE);
                    // Clear the audio buffer
                    self.audio.clear();
                }

                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                continue;
            }

            t_last = t_now;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            // TODO: Non-Vad implementation -> Bring these back.
            // let token_buffer = self.token_buffer.clone();
            // Self::set_full_params(&mut params, &self.configs, Some(&token_buffer));
            Self::set_full_params(&mut params, &self.configs, None);

            let new_audio_samples = whisper_rs::convert_stereo_to_mono_audio(&audio_samples)
                .expect("failed to convert to mono");

            let result = whisper_state.full(params, &new_audio_samples);

            if let Err(e) = result {
                match e {
                    //TODO: This can/should be removed; the model doesn't run if there's no voice.
                    WhisperError::NoSamples => {
                        eprintln!("no samples");
                        std::io::stdout().flush().unwrap();
                        t_last = t_now;
                        continue;
                    }
                    _ => {
                        self.data_sender
                            .send(Err(
                            WhisperRealtimeError::new(WhisperRealtimeErrorType::TranscriptionError, String::from("Model Failure"))

                            ))
                            .expect("Failed to send error");
                    }
                }
            }

            let num_segments = whisper_state
                .full_n_segments()
                .expect("failed to get segments");
            if num_segments == 0 {
                continue;
            }
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

            let push_new_audio = phrase_finished || self.output_buffer.is_empty();

            if push_new_audio {
                self.output_buffer.push(text_string.clone());
                phrase_finished = false;
            } else {
                let last_index = self.output_buffer.len() - 1;
                self.output_buffer[last_index] = text_string.clone();
            }

            // Send the new text to the G/UI.
            self.data_sender
                .send(Ok((text_string, push_new_audio)))
                .expect("Failed to send transcription");

            // set the flag if over timeout.

            if total_time > self.configs.realtime_timeout {

            }
        }

        self.output_buffer.join("").clone()
    }

    // TODO: Refactor this code back in to parameterize VAD use.
    //
    //
    // Keep a small amount of audio data for word boundaries.
    // let keep_from =
    //     std::cmp::max(0, self.audio_buffer.len() - constants::N_SAMPLES_KEEP - 1);
    // let keep_from = if self.audio_buffer.len() > (constants::N_SAMPLES_KEEP - 1) {
    //     self.audio_buffer.len() - constants::N_SAMPLES_KEEP - 1
    // } else {
    //     0
    // };
    // self.audio_buffer = self.audio_buffer.drain(keep_from..).collect();
    //
    // // Seed the next prompt:
    // // Note: if setting no_context, this needs to be skipped
    // let mut tokens: Vec<WhisperToken> = vec![];
    // for i in 0..num_segments {
    //     let token_count = state.full_n_tokens(i).expect("tokens failed");
    //     for j in 0..token_count {
    //         let token = state.full_get_token_id(i, j).expect("failed to get token");
    //         tokens.push(token);
    //     }
    // }
    //
    // let new_tokens: Vec<std::ffi::c_int> = tokens
    //     .into_iter()
    //     .map(|token| token as std::ffi::c_int)
    //     .collect();
    //
    // self.token_buffer = new_tokens;
}
