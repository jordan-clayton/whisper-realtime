use std::collections::VecDeque;
use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};
use std::thread::sleep;
use std::time::{Duration, Instant};

use strsim::jaro_winkler;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::transcriber::vad::VAD;
use crate::transcriber::{
    Transcriber, TranscriptionSnapshot, WhisperControlPhrase, WhisperOutput, WhisperSegment,
};
use crate::utils::constants;
use crate::utils::errors::RibbleWhisperError;
use crate::utils::Sender;
use crate::whisper::configs::WhisperRealtimeConfigs;

/// Builder for [RealtimeTranscriber]
/// All fields are necessary and thus required to successfully build a RealtimeTranscriber.
/// Multiple VAD implementations have been provided, see: [crate::transcriber::vad]
/// Silero: [crate::transcriber::vad::Silero] is recommended for accuracy.
/// See: examples/realtime_transcriber.rs for example usage.
pub struct RealtimeTranscriberBuilder<V: VAD<f32>> {
    configs: Option<WhisperRealtimeConfigs>,
    audio_buffer: Option<AudioRingBuffer<f32>>,
    output_sender: Option<Sender<WhisperOutput>>,
    voice_activity_detector: Option<V>,
}

impl<V: VAD<f32>> RealtimeTranscriberBuilder<V> {
    pub fn new() -> Self {
        Self {
            configs: None,
            audio_buffer: None,
            output_sender: None,
            voice_activity_detector: None,
        }
    }

    /// Set configurations.
    pub fn with_configs(mut self, configs: WhisperRealtimeConfigs) -> Self {
        self.configs = Some(configs);
        self
    }
    /// Set the (shared) AudioRingBuffer.
    pub fn with_audio_buffer(mut self, audio_buffer: &AudioRingBuffer<f32>) -> Self {
        self.audio_buffer = Some(audio_buffer.clone());
        self
    }

    /// Set the output sender.
    pub fn with_output_sender(mut self, sender: Sender<WhisperOutput>) -> Self {
        self.output_sender = Some(sender);
        self
    }

    /// Set the voice activity detector.
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

    /// This returns a tuple struct containing both the transcriber object and a handle to check the
    /// transcriber's ready state from another location.
    /// Returns Err when a parameter is missing.
    pub fn build(
        self,
    ) -> Result<(RealtimeTranscriber<V>, RealtimeTranscriberHandle), RibbleWhisperError> {
        let configs = self.configs.ok_or(RibbleWhisperError::ParameterError(
            "Configs missing in RealtimeTranscriberBuilder.".to_string(),
        ))?;
        let audio_feed = self.audio_buffer.ok_or(RibbleWhisperError::ParameterError(
            "Audio feed missing in RealtimeTranscriberBuilder".to_string(),
        ))?;
        let output_sender = self
            .output_sender
            .ok_or(RibbleWhisperError::ParameterError(
                "Output sender missing in RealtimeTranscriberBuilder".to_string(),
            ))?;
        let vad = self
            .voice_activity_detector
            .ok_or(RibbleWhisperError::ParameterError(
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

/// A realtime whisper transcription runner. See: examples/realtime_stream.rs for suggested use
/// RealtimeTranscriber cannot be shared across threads because it has a singular ready state.
/// It is also infeasible to call [Transcriber::process_audio] in parallel due
/// to the cost of running whisper.
pub struct RealtimeTranscriber<V: VAD<f32>> {
    configs: WhisperRealtimeConfigs,
    /// The shared input buffer from which samples are pulled for transcription
    audio_feed: AudioRingBuffer<f32>,
    /// For sending output to a UI
    output_sender: Sender<WhisperOutput>,
    /// Ready flag.
    /// A RealtimeTranscriber is considered to be ready when all of its whisper initialization has completed,
    /// and it is about to enter its transcription loop.
    /// NOTE: This cannot be accessed directly, because RealtimeTranscriber is not Sync.
    /// Use a [RealtimeTranscriberHandle] to check the ready state.
    ready: Arc<AtomicBool>,
    /// For voice detection
    vad: V,
}

impl<V: VAD<f32>> Transcriber for RealtimeTranscriber<V> {
    // This streaming implementation uses a sliding window + VAD + diffing approach to approximate
    // a continuous audio file. This will only start transcribing segments when voice is detected.
    // Its accuracy isn't bulletproof (and highly depends on the model), but it's reasonably fast
    // on average hardware.
    // GPU processing is more or less a necessity for running realtime; this will not work well using CPU inference.
    //
    // This implementation is synchronous and can be run on a single thread--however, due to the
    // bounded channel, it is recommended to process in parallel/spawn a worker to drain the data channel
    //
    // Argument:
    // - run_transcription: an atomic state flag so that the transcriber can be terminated from another location
    // eg. UI
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, RibbleWhisperError> {
        // Alert the UI
        self.output_sender
            .send(WhisperOutput::ControlPhrase(
                WhisperControlPhrase::GettingReady,
            ))
            .map_err(|e| RibbleWhisperError::TranscriptionSenderError(e.0.into_inner()))?;
        let mut t_last = Instant::now();

        // To collect audio from the ring buffer.
        let mut audio_samples: Vec<f32> = vec![0f32; constants::N_SAMPLES_30S];

        // For timing the transcription (and timeout)
        let mut total_time = 0u128;

        // For collecting the transcribed segments to return a full transcription at the end
        let mut output_string = String::default();
        let mut working_set: VecDeque<WhisperSegment> =
            VecDeque::with_capacity(constants::WORKING_SET_SIZE);

        // Set up whisper
        let full_params = self.configs.to_whisper_full_params();
        let whisper_context_params = self.configs.to_whisper_context_params();
        let file_path_string = self.configs.model().file_path_string()?;

        let ctx =
            whisper_rs::WhisperContext::new_with_params(&file_path_string, whisper_context_params)?;

        let mut whisper_state = ctx.create_state()?;
        self.ready.store(true, Ordering::Release);
        self.output_sender
            .send(WhisperOutput::ControlPhrase(
                WhisperControlPhrase::StartSpeaking,
            ))
            .map_err(|e| RibbleWhisperError::TranscriptionSenderError(e.0.into_inner()))?;
        while run_transcription.load(Ordering::Acquire) {
            let t_now = Instant::now();
            let diff = t_now - t_last;
            let millis = diff.as_millis();
            total_time += millis;

            // To prevent accidental audio clearing, hold off to ensure at least
            // vad_sample_len ms have passed before trying to detect voice.
            if millis < self.configs.vad_sample_len() as u128 {
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                continue;
            }

            // read_into will return min(requested_len, audio_len)
            // It will also escape early if the buffer is length 0
            self.audio_feed
                .read_into(self.configs.vad_sample_len(), &mut audio_samples);

            let vad_size = (self.configs.vad_sample_len() as f64 / 1000f64
                * constants::WHISPER_SAMPLE_RATE) as usize;

            // If the buffer has recently been cleared/there's not enough data to send to the voice detector,
            // sleep for a little bit longer.
            if audio_samples.len() < vad_size {
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                continue;
            }

            // Check for voice activity
            let voice_detected = self.vad.voice_detected(&audio_samples);
            if !voice_detected {
                // Drain the dequeue and push to the confirmed output_string
                let next_output = working_set.drain(..).map(|output| output.text);
                output_string.extend(next_output);

                // Clear the audio buffer to prevent data incoherence messing up the transcription.
                self.audio_feed.clear();

                // Sleep for a little bit to give the buffer time to fill up
                sleep(Duration::from_millis(constants::PAUSE_DURATION));
                // Jump to the next iteration.
                continue;
            }

            // Update the time (for timeout)
            t_last = t_now;

            // Read the audio buffer in chunks of audio_sample_len
            self.audio_feed
                .read_into(self.configs.audio_sample_len_ms(), &mut audio_samples);

            let _ = whisper_state.full(full_params.clone(), &audio_samples)?;
            let num_segments = whisper_state.full_n_segments()?;
            if num_segments == 0 {
                continue;
            }

            // It's not a big deal if there are invalid utf-8 characters
            // Use lossy to just swap it with the replacement character
            let mut segments = (0..num_segments)
                .map(|i| {
                    let text = whisper_state.full_get_segment_text_lossy(i)?;
                    let start_time = whisper_state.full_get_segment_t0(i)? * 10;
                    let end_time = whisper_state.full_get_segment_t1(i)? * 10;
                    Ok(WhisperSegment {
                        text,
                        start_time,
                        end_time,
                    })
                })
                .collect::<Result<Vec<WhisperSegment>, whisper_rs::WhisperError>>()?;

            // If the working set is empty, push the segments into the working set.
            // ie. This should only happen on first run.
            if working_set.is_empty() {
                working_set.extend(segments.drain(..));
            }
            // Otherwise, run the diffing algorithm
            // This is admittedly a little haphazard, but it seems good enough and can tolerate
            // long sentences reasonably well. It is likely that a phrase will finish and get detected
            // by the VAD well before issues are encountered.
            // There are small chances of false negatives (duplicated output), and false positives (clobbering)
            // These tend to happen with abnormal speech patterns (extra long sentence length), strange prosody, and the like.

            // In the worst cases, the entire working set can decay, but it is rare and very difficult to trigger
            // because of how whisper works.
            else {
                // Run a diff over the last N_SEGMENTS of the working set and the new segments and try
                // to resolve overlap.
                let old_segments = working_set.make_contiguous();
                // Get the tail N_SEGMENTS
                let old_len = old_segments.len();
                let tail =
                    old_segments[old_len.saturating_sub(constants::N_SEGMENTS_DIFF)..].iter_mut();

                let head = segments
                    .drain(..constants::N_SEGMENTS_DIFF.min(segments.len()))
                    .collect::<Vec<_>>();

                let mut swap_indices: Vec<_> = vec![];
                // For collecting swaps
                swap_indices.reserve(constants::N_SEGMENTS_DIFF);

                // This is Amortized O(1), with an upper bound of constants::N_SEGMENTS_DIFF * constants::N_SEGMENTS_DIFF iterations
                // In practice this is very unlikely to hit that upper bound.
                for old_segment in tail {
                    // This might be a little conservative, but it's better to be safe.
                    // It is expected that if there is a good match, it's 1:1 on each of the timestamps
                    let mut best_score = 0.0;
                    let mut best_match = None::<&WhisperSegment>;
                    // This is out of bounds, but it should always be swapped if there's a best_match
                    let mut best_index = constants::N_SEGMENTS_DIFF;
                    // Get the head N_SEGMENTS
                    for (index, new_segment) in head.iter().enumerate() {
                        // With the way that segments are being output, it seems to work a little better
                        // If when comparing timestamps, to match on starting alignment.
                        // The size of the working set is small enough such that the
                        let time_gap = (old_segment.start_time - new_segment.start_time).abs();
                        let old_lower = old_segment.text.to_lowercase();
                        let new_lower = new_segment.text.to_lowercase();
                        let similar = jaro_winkler(&old_lower, &new_lower);
                        // If it's within the same alignment, it's likely for the segments to be
                        // a match (ie. the audio buffer has recently been cleared, and this is a new window)
                        // Compare based on similarity to confirm.

                        // If the timestamp is close enough such that it's lower than the epsilon: (10 ms)
                        // Consider it to be a 1:1 match.
                        let timestamp_close = time_gap < constants::TIMESTAMP_EPSILON
                            && similar > constants::DIFF_THRESHOLD_MIN;
                        let compare_score = if time_gap <= constants::TIMESTAMP_GAP {
                            timestamp_close || {
                                if similar >= constants::DIFF_THRESHOLD_MED {
                                    true
                                } else if similar >= constants::DIFF_THRESHOLD_LOW {
                                    best_score < constants::DIFF_THRESHOLD_MED
                                        || best_match.is_none()
                                } else if similar > constants::DIFF_THRESHOLD_MIN {
                                    best_score < constants::DIFF_THRESHOLD_LOW
                                        || best_match.is_none()
                                } else {
                                    false
                                }
                            }
                        } else {
                            // Otherwise, if it's outside the timestamp gap, only match on likely probability
                            // High matches indicate close segments (ie. a word/phrase boundary)
                            if similar >= constants::DIFF_THRESHOLD_HIGH {
                                true
                            } else if similar >= constants::DIFF_THRESHOLD_MED {
                                best_score < constants::DIFF_THRESHOLD_HIGH || best_match.is_none()
                            } else if similar >= constants::DIFF_THRESHOLD_LOW {
                                best_score < constants::DIFF_THRESHOLD_MED || best_match.is_none()
                            } else {
                                false
                            }
                        };

                        if compare_score && similar > best_score {
                            best_score = similar;
                            best_match = Some(new_segment);
                            best_index = index;
                        }
                    }
                    if let Some(new_seg) = best_match {
                        // Swap the longer of the two segments in hopes that false positives do not clobber the output
                        // And also that it is the most semantically correct output. Anticipate and expect a little crunchiness.
                        if new_seg.text.len() > old_segment.text.len() {
                            *old_segment = new_seg.clone();
                        }
                        assert_ne!(best_index, constants::N_SEGMENTS_DIFF);
                        swap_indices.push(best_index)
                    }
                }

                // This is Amortized O(1), with an upper bound of constants::N_SEGMENTS_DIFF * constants::N_SEGMENTS_DIFF iterations
                // In practice this is very unlikely to hit that upper bound.
                for (index, new_seg) in head.into_iter().enumerate() {
                    if !swap_indices.contains(&index) {
                        // If the segment was never swapped in, treat it as a new segment and push it to the output buffer
                        // It is not impossible for this to break sequence, but it is highly unlikely.
                        working_set.push_back(new_seg)
                    }
                }
            }
            // If there are any remaining segments, drain them into the working set.
            working_set.extend(segments.drain(..));

            // Drain the working set when it exceeds its bounded size. It is most likely that the
            // n segments drained are actually part of the transcription.
            // It is highly, highly unlikely for this condition to ever trigger, given that VAD are
            // generally pretty good at detecting pauses.
            // It is most likely that the working set will get drained beforehand, but this is a
            // fallback to ensure the working_set is always WORKING_SET_SIZE
            if working_set.len() > constants::WORKING_SET_SIZE {
                let next_text = working_set
                    .drain(
                        0..working_set
                            .len()
                            .saturating_sub(constants::WORKING_SET_SIZE),
                    )
                    .map(|segment| segment.text);
                // Push to the output string.
                output_string.extend(next_text);
            }

            // Send the current transcription as it exists, so that the UI can update
            if !output_string.is_empty() || !working_set.is_empty() {
                let snapshot = Arc::new(TranscriptionSnapshot::new(
                    output_string.clone(),
                    working_set
                        .iter()
                        .map(|segment| segment.text.clone())
                        .collect(),
                ));
                self.output_sender
                    .send(WhisperOutput::TranscriptionSnapshot(snapshot))
                    .map_err(|e| RibbleWhisperError::TranscriptionSenderError(e.0.into_inner()))?
            }

            // If the timeout is set to 0, this loop runs infinitely.
            if self.configs.realtime_timeout() == 0 {
                continue;
            }

            // Otherwise check for timeout.
            if total_time > self.configs.realtime_timeout() {
                self.output_sender
                    .send(WhisperOutput::ControlPhrase(
                        WhisperControlPhrase::TranscriptionTimeout,
                    ))
                    .map_err(|e| RibbleWhisperError::TranscriptionSenderError(e.0.into_inner()))?;
                run_transcription.store(false, Ordering::Release);
            }
        }

        self.output_sender
            .send(WhisperOutput::ControlPhrase(
                WhisperControlPhrase::EndTranscription,
            ))
            .map_err(|e| RibbleWhisperError::TranscriptionSenderError(e.0.into_inner()))?;

        // Clean up the whisper context
        drop(whisper_state);
        drop(ctx);

        // Drain the last of the working set.
        let next_text = working_set.drain(..).map(|segment| segment.text);
        output_string.extend(next_text);

        // Set internal state to non-ready in case the transcriber is going to be reused
        self.ready.store(false, Ordering::Release);
        // Strip remaining whitespace and return
        Ok(output_string.trim().to_string())
    }
}

/// A simple handle that allows for checking the ready state of a RealtimeTranscriber from another
/// location (eg. a different thread).
#[derive(Clone)]
pub struct RealtimeTranscriberHandle {
    ready: Arc<AtomicBool>,
}

impl RealtimeTranscriberHandle {
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}
