use std::sync::{Arc, atomic::AtomicBool, atomic::Ordering};
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;

use voice_activity_detector::VoiceActivityDetector;
use whisper_rs::{FullParams, SamplingStrategy, WhisperState};

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::transcriber::{traits::Transcriber, vad, vad::VoiceActivityDetection};
use crate::utils::constants;
use crate::utils::errors::WhisperRealtimeError;
use crate::whisper::configs::WhisperConfigsV1;

// This implementation is a modified port of the whisper.cpp stream example, see:
// https://github.com/ggerganov/whisper.cpp/tree/master/examples/stream
// Realtime on CPU has not yet been tested and may or may not be feasible.
// Building with GPU support is currently recommended.

// TODO: clean up implementation, encapsulate whisper context and setup logic.
#[cfg(not(feature = "crossbeam"))]
pub struct RealtimeTranscriber {
    configs: Arc<WhisperConfigsV1>,
    audio: Arc<AudioRingBuffer<f32>>,
    output_buffer: Vec<String>,
    data_sender: mpsc::Sender<Result<(String, bool), WhisperRealtimeError>>,
    vad: Option<VoiceActivityDetector>,
}

#[cfg(feature = "crossbeam")]
pub struct RealtimeTranscriber {
    configs: Arc<WhisperConfigsV1>,
    audio: Arc<AudioRingBuffer<f32>>,
    output_buffer: Vec<String>,
    data_sender: crossbeam::channel::Sender<Result<(String, bool), WhisperRealtimeError>>,
    vad: Option<VoiceActivityDetector>,
}

impl RealtimeTranscriber {
    #[cfg(not(feature = "crossbeam"))]
    pub fn new(
        audio: Arc<AudioRingBuffer<f32>>,
        data_sender: mpsc::Sender<Result<(String, bool), WhisperRealtimeError>>,
        vad_strategy: Option<vad::VadStrategy>,
    ) -> Self {
        let strategy = vad_strategy.unwrap_or(vad::VadStrategy::default());
        let output_buffer: Vec<String> = vec![];
        let vad = Self::init_vad(strategy);

        Self {
            configs: Arc::new(WhisperConfigsV1::default()),
            audio,
            output_buffer,
            data_sender,
            vad,
        }
    }

    #[cfg(feature = "crossbeam")]
    pub fn new(
        audio: Arc<AudioRingBuffer<f32>>,
        data_sender: crossbeam::channel::Sender<Result<(String, bool), WhisperRealtimeError>>,
        vad_strategy: Option<vad::VadStrategy>,
    ) -> Self {
        let strategy = vad_strategy.unwrap_or(vad::VadStrategy::default());
        let output_buffer: Vec<String> = vec![];
        let vad = Self::init_vad(strategy);

        Self {
            configs: Arc::new(WhisperConfigsV1::default()),
            audio,
            output_buffer,
            data_sender,
            vad,
        }
    }

    #[cfg(not(feature = "crossbeam"))]
    pub fn new_with_configs(
        audio: Arc<AudioRingBuffer<f32>>,
        data_sender: mpsc::Sender<Result<(String, bool), WhisperRealtimeError>>,
        configs: Arc<WhisperConfigsV1>,
        vad_strategy: Option<vad::VadStrategy>,
    ) -> Self {
        let strategy = vad_strategy.unwrap_or(vad::VadStrategy::default());
        let output_buffer: Vec<String> = vec![];
        let vad = Self::init_vad(strategy);
        Self {
            configs,
            audio,
            output_buffer,
            data_sender,
            vad,
        }
    }

    #[cfg(feature = "crossbeam")]
    pub fn new_with_configs(
        audio: Arc<AudioRingBuffer<f32>>,
        data_sender: crossbeam::channel::Sender<Result<(String, bool), WhisperRealtimeError>>,
        configs: Arc<WhisperConfigsV1>,
        vad_strategy: Option<vad::VadStrategy>,
    ) -> Self {
        let strategy = vad_strategy.unwrap_or(vad::VadStrategy::default());
        let output_buffer: Vec<String> = vec![];
        let vad = Self::init_vad(strategy);
        Self {
            configs,
            audio,
            output_buffer,
            data_sender,
            vad,
        }
    }

    fn init_vad(strategy: vad::VadStrategy) -> Option<VoiceActivityDetector> {
        match strategy {
            vad::VadStrategy::Naive => None,
            vad::VadStrategy::Silero => Some(
                VoiceActivityDetector::builder()
                    .sample_rate(constants::WHISPER_SAMPLE_RATE as i64)
                    .chunk_size(1024usize)
                    .build()
                    .expect("failed to build voice activity detector"),
            ),
        }
    }
}

impl<
        T: voice_activity_detector::Sample
            + num_traits::cast::FromPrimitive
            + num_traits::cast::ToPrimitive
            + num_traits::cast::NumCast
            + num_traits::Zero,
    > VoiceActivityDetection<T> for RealtimeTranscriber
{
}

impl Transcriber for RealtimeTranscriber {
    fn process_audio(
        &mut self,
        whisper_state: &mut WhisperState,
        run_transcription: Arc<AtomicBool>,
        _: Option<impl FnMut(i32) + Send + Sync + 'static>,
    ) -> String {
        let mut t_last = std::time::Instant::now();

        let mut audio_samples: Vec<f32> = vec![0f32; constants::N_SAMPLES_30S];
        let mut audio_samples_vad: Vec<f32> = vec![0f32; constants::N_SAMPLES_30S];

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        Self::set_full_params(&mut params, &self.configs, None);

        let mut pause_detected = true;
        let mut phrase_finished = false;

        let mut total_time: u128 = 0;

        self.data_sender
            .send(Ok((String::from("[START SPEAKING]\n"), true)))
            .expect("Failed to send transcription");

        loop {
            if !run_transcription.load(Ordering::Acquire) {
                self.data_sender
                    .send(Ok((String::from("\n[END TRANSCRIPTION]\n"), true)))
                    .expect("Failed to send transcription");
                break;
            }

            let t_now = std::time::Instant::now();

            let diff = t_now - t_last;
            let millis = diff.as_millis();
            total_time += millis;

            if millis < self.configs.vad_sample_ms as u128 {
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                continue;
            }

            if millis >= self.configs.phrase_timeout as u128 {
                phrase_finished = true;
            }

            self.audio
                .get_audio(self.configs.vad_sample_ms, &mut audio_samples_vad);

            // TODO: refactor this once VAD has been refactored
            let check_voice_detected = match &mut self.vad {
                None => Self::is_voice_detected_naive(
                    constants::WHISPER_SAMPLE_RATE,
                    &mut audio_samples_vad,
                    // &mut new_samples,
                    self.configs.naive_vad_energy_threshold,
                    self.configs.naive_window_len,
                    self.configs.naive_window_step,
                    self.configs.naive_vad_freq_threshold,
                    self.configs.voice_probability_threshold,
                ),
                Some(ref mut vad) => Ok(Self::is_voice_detected_silero(
                    vad,
                    &mut audio_samples_vad,
                    // &mut new_samples,
                    self.configs.voice_probability_threshold,
                )),
            };

            match check_voice_detected {
                Ok(detected) => {
                    if detected {
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

                            self.audio.clear();
                        }

                        sleep(Duration::from_millis(constants::PAUSE_DURATION));
                        continue;
                    }
                }
                Err(e) => {
                    self.data_sender
                        .send(Err(e))
                        .expect("Data channel should be open");
                    continue;
                }
            }

            t_last = t_now;
            // TODO: correct this. This doesn't make a lot of sense; the configurations do not change during runtime, as far as I remember.
            // And if they do, they shouldn't
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            Self::set_full_params(&mut params, &self.configs, None);

            let new_audio_samples = whisper_rs::convert_stereo_to_mono_audio(&audio_samples)
                .expect("failed to convert to mono");

            let result = whisper_state.full(params, &new_audio_samples);

            if let Err(e) = result {
                self.data_sender
                    // It makes more sense as a function that returns Whisper FullParams
                    .send(Err(WhisperRealtimeError::TranscriptionError(format!(
                        "WhisperError: {}",
                        e
                    ))))
                    .expect("Data channel should be open");
                continue;
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

            // Set the flag if over timeout.
            if self.configs.realtime_timeout > 0 {
                if total_time > self.configs.realtime_timeout {
                    self.data_sender
                        .send(Ok((String::from("\n[TRANSCRIPTION TIMEOUT\n"), true)))
                        .expect("Failed to send transcription data");
                    run_transcription.store(false, Ordering::Release);
                }
            }
        }

        self.output_buffer.join("").clone()
    }
}
