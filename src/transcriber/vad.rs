use parking_lot::Mutex;
use voice_activity_detector::{IteratorExt, LabeledAudio};

use crate::audio::pcm::PcmS16Convertible;
use crate::utils::constants;
use crate::utils::errors::WhisperRealtimeError;

/// TODO: properly document this trait to explain why it's here, what it's for, and what's available
/// eg. Three provided VAD implementations: Silero, WebRtc, Earshot (a WebRtc) impl
/// tl;dr:
/// Silero is accurate and efficient; it's fast enough for realtime.
/// WebRtc is extremely fast and is accurate-enough; it's built for realtime.
/// Earshot is marginally faster than WebRtc and has poor accuracy; use as a fallback on lower hardware;
/// as far as I know, it's much less accurate, but it is very fast--YMMV.
pub trait VAD<T>: Resettable {
    // For realtime VAD to determine pauses, ends of phrases, and to reduce the amount of whisper
    // processing.
    fn voice_detected(&mut self, samples: &[T]) -> bool;
    // For optimizing offline transcription by reducing the amount of audio that whisper needs to process
    fn extract_voiced_frames(&mut self, samples: &[T]) -> Box<[T]>;
}

pub trait Resettable {
    fn reset_session(&mut self);
}

/// A basic builder that produces a Silero VAD backend for use in realtime transcription.
/// In effect, this adapts voice_activity_detector's builder and includes a starting detection probability.
/// The probability threshold can be swapped after building a Silero VAD backend if needed.
/// To mainatain the same flexibility as the supporting library, the sample rates and chunk sizes
/// are not constrained. Their limitations are mentioned below:
///
/// See: https://docs.rs/voice_activity_detector/0.2.0/voice_activity_detector/index.html#standalone-voice-activity-detector
/// The provided model is trained using chunk sizes of 256, 512, and 768 samples for an 8kHz sample rate
/// It is also trained using chunk sizes of 512, 768, and 1024 for a 16kHz sample rate.
/// These are not hard-requirements, but are recommended for performance
/// The only hard requirement is that the sample rate must be no larger than 31.25 times the chunk size.
/// NOTE: detection_probability_threshold is considered to be a lower bound and is noninclusive;
/// computed probabilities higher than this threshold are considered to have detected some amount
/// of voice activity.
#[derive(Copy, Clone)]
pub struct SileroBuilder {
    sample_rate: i64,
    chunk_size: usize,
    detection_probability_threshold: f32,
}

impl SileroBuilder {
    pub fn new() -> Self {
        Self {
            sample_rate: 0,
            chunk_size: 0,
            detection_probability_threshold: 0.,
        }
    }
    pub fn with_sample_rate(mut self, sample_rate: i64) -> Self {
        self.sample_rate = sample_rate;
        self
    }
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
    pub fn with_detection_probability_threshold(mut self, probability: f32) -> Self {
        self.detection_probability_threshold = probability;
        self
    }

    // It's not entirely clear as to what might cause VoiceActivityDetectorBuilder::build() to fail
    // but this function returns Err if the Silero Vad struct fails to build.
    pub fn build(self) -> Result<Silero, WhisperRealtimeError> {
        voice_activity_detector::VoiceActivityDetector::builder()
            .sample_rate(self.sample_rate)
            .chunk_size(self.chunk_size)
            .build()
            .map(|vad| Silero {
                vad,
                detection_probability_threshold: self.detection_probability_threshold,
            })
            .map_err(|e| {
                WhisperRealtimeError::ParameterError(format!(
                    "Failed to build Silero VAD. Error: {}",
                    e
                ))
            })
    }
}

/// Represents the Silero VAD backend for use in realtime transcription
/// Adapts voice_activity_detector::VoiceActivityDetector to predict voice activity using
/// Silero and the ONNX runtime.
/// NOTE: on Windows, this may or may not have telemetry, see: https://docs.rs/ort/latest/ort/#strategies
/// If this is a problem, please file an issue.
pub struct Silero {
    vad: voice_activity_detector::VoiceActivityDetector,
    detection_probability_threshold: f32,
}

impl Silero {
    pub fn with_detection_probability_threshold(mut self, probability: f32) -> Self {
        self.detection_probability_threshold = probability;
        self
    }

    /// A "Default" whisper-ready Silero configuration
    pub fn try_new_whisper_realtime_default() -> Result<Self, WhisperRealtimeError> {
        SileroBuilder::new()
            .with_sample_rate(constants::WHISPER_SAMPLE_RATE as i64)
            .with_chunk_size(constants::SILERO_CHUNK_SIZE)
            .with_detection_probability_threshold(constants::SILERO_VOICE_PROBABILITY_THRESHOLD)
            .build()
    }

    pub fn try_new_whisper_offline_default() -> Result<Self, WhisperRealtimeError> {
        SileroBuilder::new()
            .with_sample_rate(constants::WHISPER_SAMPLE_RATE as i64)
            .with_chunk_size(constants::SILERO_CHUNK_SIZE)
            .with_detection_probability_threshold(constants::OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .build()
    }
}

impl Resettable for Silero {
    fn reset_session(&mut self) {
        // VoiceActivityDetector does not reset configurations to default settings when resetting the context
        // so this method is just a simple delegate.
        self.vad.reset()
    }
}

impl<T: voice_activity_detector::Sample> VAD<T> for Silero {
    /// NOTE: This implementation assumes that the samples are at the same sample rate as the configured VAD
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        // If a zero-length slice of samples are sent, there is obviously no voice
        if samples.len() == 0 {
            return false;
        }
        // Create a LabelIterator to stream the prediction over the sample chunks at the given detection threshold.
        // LabelIterator allows for padding to compensate for sudden speech cutoffs/gaps in audio;
        // 3 frames is expected to be sufficient.
        let probabilites = samples.iter().copied().label(
            &mut self.vad,
            self.detection_probability_threshold,
            3usize,
        );

        // Since LabelIterator/ProbabilityIterator do not have a non-consuming way to compare
        // the number of voiced frames versus the total number of frames, the accmumulation has to
        // be done explicitly.
        let mut total_frames = 0usize;
        let mut voiced_frames = 0usize;
        for label in probabilites {
            match label {
                LabeledAudio::Speech(_) => {
                    total_frames += 1;
                    voiced_frames += 1
                }
                LabeledAudio::NonSpeech(_) => total_frames += 1,
            }
        }

        assert_ne!(total_frames, 0);
        // If more than half the frames meet the given threshold, treat the sample as containing speech
        let voiced_proportion = voiced_frames as f32 / total_frames as f32;
        voiced_proportion > 0.5
    }
    fn extract_voiced_frames(&mut self, samples: &[T]) -> Box<[T]> {
        if samples.len() == 0 {
            return vec![].into_boxed_slice();
        }

        samples
            .iter()
            .copied()
            .label(&mut self.vad, self.detection_probability_threshold, 3usize)
            .filter(|frame| frame.is_speech())
            // Extract the chunks which contain speech
            .map(|frame| frame.iter().copied().collect::<Vec<T>>())
            .flatten()
            .collect()
    }
}

/// Wrapper Enumeration for WebRtcImplementations
/// These are intended for the WebRTCBuilder which handles constructing the desired
/// WebRtc backend.
#[derive(Copy, Clone, Debug)]
pub enum WebRtcSampleRate {
    R8kHz,
    R16kHz,
    R32kHz,
    R48kHz,
}

impl WebRtcSampleRate {
    fn to_webrtc_sample_rate(&self) -> webrtc_vad::SampleRate {
        match self {
            Self::R8kHz => webrtc_vad::SampleRate::Rate8kHz,
            Self::R16kHz => webrtc_vad::SampleRate::Rate16kHz,
            Self::R32kHz => webrtc_vad::SampleRate::Rate32kHz,
            Self::R48kHz => webrtc_vad::SampleRate::Rate48kHz,
        }
    }
    fn to_sample_rate_hz(&self) -> usize {
        match self {
            Self::R8kHz => 8000usize,
            Self::R16kHz => 16000usize,
            Self::R32kHz => 32000usize,
            Self::R48kHz => 48000usize,
        }
    }
}

// TODO: reword, implementation can and will be generalized for both WebRTC implementors.
/// Wrapper Enumeration for webrtc_vad::VadMode and earshot::VoiceActivityProfile.
/// These are intended for the WebRTCBuilder which handles constructing the desired WebRtc backend.
///
/// This parameter sets the "mode" from which predetermined speech threshold constants are selected for filtering out non-speech.
/// See: https://chromium.googlesource.com/external/webrtc/+/refs/heads/master/common_audio/vad/vad_core.c#68
/// Quality = low filtering, detects most speech and then some. May introduce some false positives
/// VeryAggressive = high filtering, only clear speech passes. Might introduce false negatives
///
/// For small samples (and clear enough audio), higher aggressiveness is likely to produce better results
#[derive(Copy, Clone, Debug)]
pub enum WebRtcFilterAggressiveness {
    Quality,
    LowBitrate,
    Aggressive,
    VeryAggressive,
}

impl WebRtcFilterAggressiveness {
    fn to_webrtc_vad_mode(&self) -> webrtc_vad::VadMode {
        match self {
            Self::Quality => webrtc_vad::VadMode::Quality,
            Self::LowBitrate => webrtc_vad::VadMode::LowBitrate,
            Self::Aggressive => webrtc_vad::VadMode::Aggressive,
            Self::VeryAggressive => webrtc_vad::VadMode::VeryAggressive,
        }
    }
    fn to_earshot_vad_profile(&self) -> earshot::VoiceActivityProfile {
        match self {
            Self::Quality => earshot::VoiceActivityProfile::QUALITY,
            Self::LowBitrate => earshot::VoiceActivityProfile::LBR,
            Self::Aggressive => earshot::VoiceActivityProfile::AGGRESSIVE,
            Self::VeryAggressive => earshot::VoiceActivityProfile::VERY_AGGRESSIVE,
        }
    }
}

/// Since WebRTC expects frames of specific fixed length, this enumeration captures the only possible
/// valid lengths. It is not used directly in the implementation or the VAD backend; its purpose is
/// to provide information required to compute the necessary sample padding/truncation to fit the
/// frame size requirements
/// This is more or less equivalent to voice_activity_detector's more flexible chunk_size parameter,
/// except measured in milliseconds instead of individual PCM samples.

#[derive(Copy, Clone, Debug)]
pub enum WebRtcFrameLengthMillis {
    MS10 = 10,
    MS20 = 20,
    MS30 = 30,
}

impl WebRtcFrameLengthMillis {
    fn to_ms(&self) -> usize {
        match self {
            Self::MS10 => 10usize,
            Self::MS20 => 20usize,
            Self::MS30 => 30usize,
        }
    }
}

/// A builder that can produce either a WebRtc or Earshot backend for use in realtime transcription.
/// Due to the way WebRTC is implemented, detection_probability_threshold should be treated as the
/// minimum proportion of frames that are detected to have speech.
/// This is a non-inclusive lower-bound; samples with VAD frame proportions higher than this
/// threshold are thus considered to contain speech
#[derive(Copy, Clone)]
pub struct WebRtcBuilder {
    sample_rate: WebRtcSampleRate,
    aggressiveness: WebRtcFilterAggressiveness,
    frame_length: WebRtcFrameLengthMillis,
    detection_probability_threshold: f32,
}

impl WebRtcBuilder {
    pub fn new() -> Self {
        Self {
            sample_rate: WebRtcSampleRate::R8kHz,
            aggressiveness: WebRtcFilterAggressiveness::Quality,
            frame_length: WebRtcFrameLengthMillis::MS10,
            detection_probability_threshold: 0.0,
        }
    }
    pub fn with_sample_rate(mut self, sample_rate: WebRtcSampleRate) -> Self {
        self.sample_rate = sample_rate;
        self
    }
    pub fn with_filter_aggressiveness(
        mut self,
        aggressiveness: WebRtcFilterAggressiveness,
    ) -> Self {
        self.aggressiveness = aggressiveness;
        self
    }
    pub fn with_detection_probability_threshold(mut self, probability_threshold: f32) -> Self {
        self.detection_probability_threshold = probability_threshold;
        self
    }
    pub fn with_frame_length_millis(mut self, frame_length: WebRtcFrameLengthMillis) -> Self {
        self.frame_length = frame_length;
        self
    }

    // This returns none if webrtc_vad panics, which will happen in the case of a memory allocation error.
    // (eg. OOM).
    pub fn build_webrtc(self) -> Result<WebRtc, WhisperRealtimeError> {
        std::panic::catch_unwind(|| {
            webrtc_vad::Vad::new_with_rate_and_mode(
                self.sample_rate.to_webrtc_sample_rate(),
                self.aggressiveness.to_webrtc_vad_mode(),
            )
        })
        .map(|vad| WebRtc {
            vad: Mutex::new(vad),
            // WebRtcSampleRate and WebRtcFilterAgressiveness both have Copy semantics
            sample_rate: self.sample_rate,
            aggressiveness: self.aggressiveness,
            frame_length_in_ms: self.frame_length.to_ms(),
            realtime_detection_probability_threshold: self.detection_probability_threshold,
        })
        .map_err(|_| {
            WhisperRealtimeError::ParameterError(
                "Failed to build WebRTC due to memory allocation error.".to_string(),
            )
        })
    }
    pub fn build_earshot(self) -> Result<Earshot, WhisperRealtimeError> {
        let vad = earshot::VoiceActivityDetector::new(self.aggressiveness.to_earshot_vad_profile());
        let predicate = match self.sample_rate {
            WebRtcSampleRate::R8kHz => earshot::VoiceActivityDetector::predict_8khz,
            WebRtcSampleRate::R16kHz => earshot::VoiceActivityDetector::predict_16khz,
            WebRtcSampleRate::R32kHz => earshot::VoiceActivityDetector::predict_32khz,
            WebRtcSampleRate::R48kHz => earshot::VoiceActivityDetector::predict_48khz,
        };
        Ok(Earshot {
            vad,
            sample_rate: self.sample_rate.to_sample_rate_hz(),
            frame_length_in_ms: self.frame_length.to_ms(),
            realtime_detection_probability_threshold: self.detection_probability_threshold,
            prediction_predicate: predicate,
        })
    }
}

// TODO: properly document this to provide relevant information, like Silero.
pub struct WebRtc {
    vad: Mutex<webrtc_vad::Vad>,
    // Since webrtc_vad does not implement Copy or Clone, use the wrapper structs.
    // These have matching methods to produce the internal enumeration equivalent as needed
    sample_rate: WebRtcSampleRate,
    aggressiveness: WebRtcFilterAggressiveness,
    // This value will always be restricted to either 10ms, 20ms, 30ms as per the Enumeration
    // used in the accompanying builder
    frame_length_in_ms: usize,
    realtime_detection_probability_threshold: f32,
}

impl WebRtc {
    pub fn with_realtime_detection_probability_threshold(mut self, probability: f32) -> Self {
        self.realtime_detection_probability_threshold = probability;
        self
    }
    /// A "Default" whisper-ready WebRtc configuration
    pub fn try_new_whisper_realtime_default() -> Result<Self, WhisperRealtimeError> {
        WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_detection_probability_threshold(constants::WEBRTC_VOICE_PROBABILITY_THRESHOLD)
            .build_webrtc()
    }

    pub fn try_new_whisper_offline_default() -> Result<Self, WhisperRealtimeError> {
        WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Aggressive)
            .with_detection_probability_threshold(constants::OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .build_webrtc()
    }
}

// WebRtc is internally guarded as far as I know, the C-API is thread safe.
unsafe impl Send for WebRtc {}
unsafe impl Sync for WebRtc {}

impl Resettable for WebRtc {
    fn reset_session(&mut self) {
        // WebRtc_vad reverts to default settings when the context is cleared.
        // The vad must be reconfigured accordingly.
        let mut vad = self.vad.lock();
        vad.reset();
        vad.set_sample_rate(self.sample_rate.to_webrtc_sample_rate());
        vad.set_mode(self.aggressiveness.to_webrtc_vad_mode());
    }
}

impl<T: PcmS16Convertible + Copy> VAD<T> for WebRtc {
    // NOTE: This implementation assumes that the samples are at the same sample rate as the configured VAD
    // Samples of insufficient length are zero-padded/truncated to avoid internal panicking.
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        // If a zero-length slice of samples are sent, there is obviously no voice
        if samples.len() == 0 {
            return false;
        }
        let (int_audio, frame_size) = prepare_webrtc_frames(
            samples,
            self.frame_length_in_ms,
            self.sample_rate.to_sample_rate_hz(),
        );

        let frames = int_audio.chunks_exact(frame_size);

        let total_num_frames = frames.len();
        assert_ne!(total_num_frames, 0);
        let mut vad = self.vad.lock();

        let speech_frames = frames.filter(|&frame| {
            // This should never, ever panic unless my arithmetic is busted
            // Unwrap to force a panic to catch errors in the implementation.
            vad.is_voice_segment(frame)
                .expect("The frame size should be valid.")
        });

        // Since WebRtc doesn't allow users to set the "threshold" directly, treat the threshold
        // like a minimum proportion of frames that have to be detected to be considered speech
        let voiced_proportion = (speech_frames.count() as f32) / (total_num_frames as f32);
        voiced_proportion > self.realtime_detection_probability_threshold
    }
    fn extract_voiced_frames(&mut self, samples: &[T]) -> Box<[T]> {
        if samples.len() == 0 {
            return vec![].into_boxed_slice();
        }
        let (int_audio, frame_size) = prepare_webrtc_frames(
            samples,
            self.frame_length_in_ms,
            self.sample_rate.to_sample_rate_hz(),
        );
        let frames = int_audio.chunks_exact(frame_size);
        let mut vad = self.vad.lock();
        frames
            .filter(|&frame| {
                vad.is_voice_segment(frame)
                    .expect("The Frame size should be valid")
            })
            .flatten()
            .map(|&s| T::from_pcm_s16(s))
            .collect()
    }
}

type EarshotPredictionFilterPredicate =
    fn(&mut earshot::VoiceActivityDetector, &[i16]) -> Result<bool, earshot::Error>;
pub struct Earshot {
    vad: earshot::VoiceActivityDetector,
    // For dispatching the appropriate method
    sample_rate: usize,
    frame_length_in_ms: usize,
    realtime_detection_probability_threshold: f32,
    prediction_predicate: EarshotPredictionFilterPredicate,
}

impl Earshot {
    pub fn with_realtime_detection_probability_threshold(mut self, probability: f32) -> Self {
        self.realtime_detection_probability_threshold = probability;
        self
    }

    /// A "Default" whisper-ready Earhshot configuration
    pub fn try_new_whisper_realtime_default() -> Result<Self, WhisperRealtimeError> {
        WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_detection_probability_threshold(constants::WEBRTC_VOICE_PROBABILITY_THRESHOLD)
            .build_earshot()
    }

    pub fn try_new_whisper_offline_default() -> Result<Self, WhisperRealtimeError> {
        WebRtcBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::Aggressive)
            .with_detection_probability_threshold(constants::OFFLINE_VOICE_PROBABILITY_THRESHOLD)
            .build_earshot()
    }
}

impl Resettable for Earshot {
    fn reset_session(&mut self) {
        // Earshot does not revert back to default settings when resetting the context,
        // so this method is just a delegate.
        self.vad.reset()
    }
}

impl<T: PcmS16Convertible + Copy> VAD<T> for Earshot {
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        // If a zero-length slice of samples are sent, there is obviously no voice
        if samples.len() == 0 {
            return false;
        }

        let (int_audio, frame_size) =
            prepare_webrtc_frames(samples, self.frame_length_in_ms, self.sample_rate);
        let frames = int_audio.chunks_exact(frame_size);

        let total_num_frames = frames.len();
        assert_ne!(total_num_frames, 0);
        let vad = &mut self.vad;
        let speech_frames = frames.filter(|&frame| {
            (self.prediction_predicate)(vad, frame).expect("Frame size should be valid.")
        });

        // Like WebRtc, (this is a WebRtc implementation),
        // doesn't allow users to set the "threshold" directly, treat the threshold
        // like a minimum proportion of frames that have to be detected to be considered speech

        let voiced_proportion = (speech_frames.count() as f32) / (total_num_frames as f32);
        voiced_proportion > self.realtime_detection_probability_threshold
    }

    fn extract_voiced_frames(&mut self, samples: &[T]) -> Box<[T]> {
        if samples.len() == 0 {
            return vec![].into_boxed_slice();
        }

        let (int_audio, frame_size) =
            prepare_webrtc_frames(samples, self.frame_length_in_ms, self.sample_rate);
        let frames = int_audio.chunks_exact(frame_size);
        let vad = &mut self.vad;
        frames
            .filter(|&frame| {
                (self.prediction_predicate)(vad, frame).expect("Frame size should be valid.")
            })
            .flatten()
            .map(|&s| T::from_pcm_s16(s))
            .collect()
    }
}

/// This small utility function handles the padding/truncation required by WebRTC implementations
/// It returns the converted and padded audio, as well as the frame size so that methods consuming
/// this function do not need to recompute the frame size.
fn prepare_webrtc_frames<T: PcmS16Convertible + Copy>(
    samples: &[T],
    frame_length_in_ms: usize,
    sample_rate: usize,
) -> (Vec<i16>, usize) {
    // Convert to integer audio
    let mut int_audio: Vec<i16> = samples.iter().map(|s| s.into_pcm_s16()).collect();
    // Because of implementation details in WebRTC, frames need to be either 10ms, 20ms, or 30ms in length
    // This means the length must be a multiple of (sample_rate * audio_length) / 1000
    // eg. 8kHz -> 80, 160, 240
    let frame_size = (sample_rate * frame_length_in_ms) / 1000;
    let zero_pad = frame_size - int_audio.len().rem_euclid(frame_size);
    int_audio.resize(int_audio.len() + zero_pad, 0);
    (int_audio, frame_size)
}
