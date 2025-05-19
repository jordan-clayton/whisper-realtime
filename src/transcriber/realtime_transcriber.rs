use std::sync::{Arc, atomic::AtomicBool, atomic::Ordering, Mutex};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::transcriber::transcriber::{Transcriber, WhisperOutput};
use crate::transcriber::vad::VAD;
use crate::utils::constants;
use crate::utils::errors::WhisperRealtimeError;
use crate::utils::sender::Sender;
use crate::whisper::configs::WhisperRealtimeConfigs;

// This implementation is a modified port of the whisper.cpp stream example, see:
// https://github.com/ggerganov/whisper.cpp/tree/master/examples/stream
// Realtime on CPU has not yet been tested and may or may not be feasible.
// Building with GPU support is currently recommended.

#[derive(Clone)]
pub struct RealtimeTranscriberBuilder<V: VAD<f32>> {
    configs: Option<Arc<WhisperRealtimeConfigs>>,
    audio_feed: Option<Arc<AudioRingBuffer<f32>>>,
    output_sender: Option<Sender<WhisperOutput>>,
    voice_activity_detector: Option<Arc<Mutex<V>>>,
}

impl<V: VAD<f32>> RealtimeTranscriberBuilder<V> {
    pub fn new() -> Self {
        Self {
            configs: None,
            audio_feed: None,
            output_sender: None,
            voice_activity_detector: None,
        }
    }
    pub fn with_configs(mut self, configs: WhisperRealtimeConfigs) -> Self {
        self.configs = Some(Arc::new(configs));
        self
    }
    pub fn with_audio_feed(mut self, audio_feed: AudioRingBuffer<f32>) -> Self {
        self.audio_feed = Some(Arc::new(audio_feed));
        self
    }
    pub fn with_output_sender(mut self, sender: Sender<WhisperOutput>) -> Self {
        self.output_sender = Some(sender);
        self
    }
    pub fn with_voice_activity_detector<U: VAD<f32>>(
        self,
        vad: U,
    ) -> RealtimeTranscriberBuilder<U> {
        let voice_activity_detector = Some(Arc::new(Mutex::new(vad)));
        RealtimeTranscriberBuilder {
            configs: self.configs,
            audio_feed: self.audio_feed,
            output_sender: self.output_sender,
            voice_activity_detector,
        }
    }

    pub fn build(self) -> Result<RealtimeTranscriber<V>, WhisperRealtimeError> {
        let configs = self.configs.ok_or(WhisperRealtimeError::ParameterError(
            "Configs missing in RealtimeTranscriberBuilder.".to_string(),
        ))?;
        let audio_feed = self.audio_feed.ok_or(WhisperRealtimeError::ParameterError(
            "Audio feed missing in RealtimeTranscriberBuilder".to_string(),
        ))?;
        let output_sender = self
            .output_sender
            .ok_or(WhisperRealtimeError::ParameterError(
                "Output sender missing in RealtimeTranscriberBuilder".to_string(),
            ))?;
        let vad = self
            .voice_activity_detector
            .ok_or(WhisperRealtimeError::ParameterError(
                "Voice activity detector missing in RealtimeTranscriberBuilder.".to_string(),
            ))?;
        Ok(RealtimeTranscriber {
            configs,
            audio_feed,
            output_sender,
            vad,
        })
    }
}

#[derive(Clone)]
pub struct RealtimeTranscriber<V: VAD<f32>> {
    configs: Arc<WhisperRealtimeConfigs>,
    audio_feed: Arc<AudioRingBuffer<f32>>,
    output_sender: Sender<WhisperOutput>,
    vad: Arc<Mutex<V>>,
}

// RealtimeTranscriber is internally protected and is therefore thread safe
// It is also trivially cloneable.
unsafe impl<V: VAD<f32>> Sync for RealtimeTranscriber<V> {}
unsafe impl<V: VAD<f32>> Send for RealtimeTranscriber<V> {}

impl<V: VAD<f32>> Transcriber for RealtimeTranscriber<V> {
    // TODO: document why run_transcription flag.
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, WhisperRealtimeError> {
        let mut t_last = Instant::now();

        // To collect audio from the ring buffer.
        let mut audio_samples: Vec<f32> = vec![0f32; constants::N_SAMPLES_30S];

        // For timing the transcription (and timeout)
        let mut total_time = 0u128;
        // For collecting the transcribed segments to return a full transcription at the end
        let mut output_buffer: Vec<String> = vec![];

        // To handle words/phrases somewhat gracefully.
        // When phrases are detected/assumed to be finished, this will send a WhisperOutput::FinishedPhrase,
        // to let the UI know to bake the output, until then, updates to the transcription will overwrite
        // what has been previously collected in attempt accurate to the collected speech
        let mut phrase_finished = true;
        // To stem the flow of newlines if/when a pause is detected. If a pause has been detected
        // and not cleared, just sleep until there's voice again.
        let mut pause_detected = true;

        // Set up whisper
        let (whisper_configs, realtime_configs) = self.configs.to_decomposed();

        let full_params = whisper_configs.to_whisper_full_params();
        let whisper_context_params = whisper_configs.to_whisper_context_params();

        let ctx = whisper_rs::WhisperContext::new_with_params(
            whisper_configs
                .model()
                .file_path()
                .to_str()
                .expect("File should be valid utf-8 str"),
            whisper_context_params,
        )?;

        let mut whisper_state = ctx.create_state()?;

        self.output_sender
            .send(WhisperOutput::FinishedPhrase(
                "[START SPEAKING]\n".to_string(),
            ))
            .map_err(|e| WhisperRealtimeError::TranscriptionSenderError(e.0.into_inner()))?;

        while run_transcription.load(Ordering::Acquire) {
            let t_now = Instant::now();
            let diff = t_now - t_last;
            let millis = diff.as_millis();
            total_time += millis;

            // If less than t=vad_sample_len() has passed, sleep for s = 100ms = constants::PAUSE_DURATION
            if millis < realtime_configs.vad_sample_len() as u128 {
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                continue;
            }
            if millis > realtime_configs.phrase_timeout() as u128 {
                phrase_finished = true;
            }

            self.audio_feed
                .read_into(realtime_configs.vad_sample_len(), &mut audio_samples);

            // Check for voice activity
            let mut vad = match self.vad.lock() {
                Ok(v) => v,
                Err(e) => {
                    self.vad.clear_poison();
                    e.into_inner()
                }
            };

            let voice_detected = vad.voice_detected(&audio_samples);
            if !voice_detected {
                // If the pause has already been detected, don't send a newline
                // Sleep for a little bit to give the buffer time to fill up
                if pause_detected {
                    sleep(Duration::from_millis(constants::PAUSE_DURATION));
                    continue;
                }

                // Otherwise, detect the pause, then send a newline.
                pause_detected = true;
                self.output_sender
                    .send(WhisperOutput::FinishedPhrase("\n".to_string()))
                    .map_err(|e| {
                        WhisperRealtimeError::TranscriptionSenderError(e.0.into_inner())
                    })?;

                // Clear the audio buffer to prevent data incoherence messing up the transcription.
                self.audio_feed.clear();

                // Sleep for a little bit to give the buffer time to fill up
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                // Jump to the next iteration.
                continue;
            }
            // Voice has been detected, finish reading and transcribing

            // First, update the time (for timeout/phrase output)
            t_last = t_now;

            let _ = whisper_state.full(full_params.clone(), &audio_samples)?;
            let num_segments = whisper_state.full_n_segments()?;
            if num_segments == 0 {
                continue;
            }

            // It's not a big deal if there are invalid utf-8 characters
            // Use lossy to just swap it with the replacement character
            let segments = (0..num_segments)
                .map(|i| whisper_state.full_get_segment_text_lossy(i))
                .collect::<Result<Vec<String>, _>>()?;

            // Join it into a full string
            let text = segments.join("");

            // Make a copy of the new text in the output buffer and send through the message channel
            if phrase_finished {
                output_buffer.push(text.clone());
                phrase_finished = false;
                self.output_sender.send(WhisperOutput::FinishedPhrase(text))
            } else {
                let last_index = output_buffer.len().saturating_sub(1);
                match output_buffer.get_mut(last_index) {
                    None => output_buffer.push(text.clone()),
                    Some(old_text) => *old_text = text.clone(),
                }
                self.output_sender
                    .send(WhisperOutput::ContinuedPhrase(text))
            }
            .map_err(|e| WhisperRealtimeError::TranscriptionSenderError(e.0.into_inner()))?;

            // If the timeout is set to 0, this loop runs infinitely.
            if realtime_configs.realtime_timeout() == 0 {
                continue;
            }

            // Otherwise check for timeout.
            if total_time > realtime_configs.realtime_timeout() {
                self.output_sender
                    .send(WhisperOutput::FinishedPhrase(
                        "\n[TRANSCRIPTION TIMEOUT]\n".to_string(),
                    ))
                    .map_err(|e| {
                        WhisperRealtimeError::TranscriptionSenderError(e.0.into_inner())
                    })?;
                run_transcription.store(false, Ordering::Release);
            }
        }
        self.output_sender
            .send(WhisperOutput::FinishedPhrase(
                "\n[END TRANSCRIPTION]\n".to_string(),
            ))
            .map_err(|e| WhisperRealtimeError::TranscriptionSenderError(e.0.into_inner()))?;

        Ok(output_buffer.join(""))
    }
}
