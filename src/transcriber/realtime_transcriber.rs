use std::sync::{Arc, atomic::AtomicBool, atomic::Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::transcriber::{Transcriber, WhisperOutput};
use crate::transcriber::vad::VAD;
use crate::utils::constants;
use crate::utils::errors::WhisperRealtimeError;
use crate::utils::Sender;
use crate::whisper::configs::WhisperRealtimeConfigs;

// This implementation is a modified port of the whisper.cpp stream example, see:
// https://github.com/ggerganov/whisper.cpp/tree/master/examples/stream
// Realtime on CPU has not yet been tested and may or may not be feasible.
// Building with GPU support is currently recommended.

pub struct RealtimeTranscriberBuilder<V: VAD<f32> + Send + Sync> {
    configs: Option<WhisperRealtimeConfigs>,
    audio_buffer: Option<Arc<AudioRingBuffer<f32>>>,
    output_sender: Option<Sender<WhisperOutput>>,
    voice_activity_detector: Option<V>,
}

impl<V: VAD<f32> + Send + Sync> RealtimeTranscriberBuilder<V> {
    pub fn new() -> Self {
        Self {
            configs: None,
            audio_buffer: None,
            output_sender: None,
            voice_activity_detector: None,
        }
    }
    pub fn with_configs(mut self, configs: WhisperRealtimeConfigs) -> Self {
        self.configs = Some(configs);
        self
    }
    // Since the AudioRingBuffer can be shared, the audio_buffer is accepted as an Arc
    pub fn with_audio_buffer(mut self, audio_buffer: Arc<AudioRingBuffer<f32>>) -> Self {
        self.audio_buffer = Some(audio_buffer);
        self
    }
    pub fn with_output_sender(mut self, sender: Sender<WhisperOutput>) -> Self {
        self.output_sender = Some(sender);
        self
    }
    pub fn with_voice_activity_detector<U: VAD<f32> + Sync + Send>(
        self,
        vad: U,
    ) -> RealtimeTranscriberBuilder<U> {
        let voice_activity_detector = Some(vad);
        RealtimeTranscriberBuilder {
            configs: self.configs,
            audio_buffer: self.audio_buffer,
            output_sender: self.output_sender,
            voice_activity_detector,
        }
    }

    // This returns a tuple struct containing both the transcriber object and a handle to check the
    // transcriber's ready state.

    // Since process_audio() runs its own infinite loop with &mut self, it's not possible to check
    // the ready state while RealtimeTranscriber is running.
    // Use RealtimeTranscriberHandle::ready() to know if the whisper context has loaded and the
    // transcribe loop has begun.
    pub fn build(
        self,
    ) -> Result<(RealtimeTranscriber<V>, RealtimeTranscriberHandle), WhisperRealtimeError> {
        let configs = self.configs.ok_or(WhisperRealtimeError::ParameterError(
            "Configs missing in RealtimeTranscriberBuilder.".to_string(),
        ))?;
        let audio_feed = self
            .audio_buffer
            .ok_or(WhisperRealtimeError::ParameterError(
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
        let ready = Arc::new(AtomicBool::new(false));

        let handle = RealtimeTranscriberHandle {
            ready: Arc::clone(&ready),
        };
        let transcriber = RealtimeTranscriber {
            configs,
            audio_feed,
            output_sender,
            ready,
            vad,
        };
        Ok((transcriber, handle))
    }
}

// TODO: proper documentation
// This cannot be Clone without introducing undefined state.
// It's also infeasible to run multiple RealtimeTranscribers in parallel due to the cost
// of running whisper transcription.
pub struct RealtimeTranscriber<V: VAD<f32> + Send + Sync> {
    configs: WhisperRealtimeConfigs,
    audio_feed: Arc<AudioRingBuffer<f32>>,
    output_sender: Sender<WhisperOutput>,
    ready: Arc<AtomicBool>,
    vad: V,
}

impl<V: VAD<f32> + Send + Sync> Transcriber for RealtimeTranscriber<V> {
    // TODO: document why run_transcription flag.
    // Also note: this can be, but probably shouldn't be called synchronously.
    // It can be called from a main thread, but the data will need to be retrieved on a
    // separate thread/asynchronously
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, WhisperRealtimeError> {
        // Spawn the thread to run the transcription.
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
        let full_params = self.configs.to_whisper_full_params();
        let whisper_context_params = self.configs.to_whisper_context_params();

        let ctx = whisper_rs::WhisperContext::new_with_params(
            self.configs.model().file_path().to_str().ok_or_else(|| {
                WhisperRealtimeError::ParameterError(format!(
                    "File, {:?} should be valid utf-8 str",
                    self.configs.model().file_path()
                ))
            })?,
            whisper_context_params,
        )?;

        let mut whisper_state = ctx.create_state()?;
        self.ready.store(true, Ordering::Release);
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
            if millis < self.configs.vad_sample_len() as u128 {
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                continue;
            }

            // TODO: fix this; it's not logically correct.
            // The phrase timeout needs to be a separate accumulator mod self.configs.phrase_timeout()
            // It might suffice to let the phrase_timeout() = audio_sample_len().
            if millis > self.configs.phrase_timeout() as u128 {
                phrase_finished = true;
            }

            self.audio_feed
                .read_into(self.configs.vad_sample_len(), &mut audio_samples);

            // Check for voice activity
            let voice_detected = self.vad.voice_detected(&audio_samples);
            if !voice_detected {
                // If the pause has already been detected, don't send a newline
                // Sleep for a little bit to give the buffer time to fill up
                if pause_detected {
                    sleep(Duration::from_millis(constants::PAUSE_DURATION));
                    continue;
                }

                // Otherwise, detect the pause, then send a newline.
                pause_detected = true;
                //Ensure the next output gets collected and sent.
                phrase_finished = true;

                // Only send a newline if the transcription is mid-transcription
                if !output_buffer.is_empty() {
                    output_buffer.push("\n".to_string());
                    self.output_sender
                        .send(WhisperOutput::FinishedPhrase("\n".to_string()))
                        .map_err(|e| {
                            WhisperRealtimeError::TranscriptionSenderError(e.0.into_inner())
                        })?;
                }

                // Clear the audio buffer to prevent data incoherence messing up the transcription.
                self.audio_feed.clear();

                // Sleep for a little bit to give the buffer time to fill up
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                // Jump to the next iteration.
                continue;
            }
            // Voice has been detected, finish reading and transcribing
            pause_detected = false;
            // First, update the time (for timeout/phrase output)
            t_last = t_now;
            // Read the buffer again
            self.audio_feed
                .read_into(self.configs.audio_sample_len(), &mut audio_samples);

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
            let text = segments.join("").trim().to_string();

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
            if self.configs.realtime_timeout() == 0 {
                continue;
            }

            // Otherwise check for timeout.
            if total_time > self.configs.realtime_timeout() {
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

        // Clean up the whisper context
        drop(whisper_state);
        drop(ctx);

        // Strip remaining whitespace
        Ok(output_buffer.join("").trim().to_string())
    }
}

#[derive(Clone)]
pub struct RealtimeTranscriberHandle {
    ready: Arc<AtomicBool>,
}

impl RealtimeTranscriberHandle {
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}
