use std::{f64::consts::PI, sync::Mutex};
use std::sync::LazyLock;

use realfft::RealFftPlanner;

use crate::audio::pcm::IntoPcmS16;
use crate::utils::constants;
use crate::utils::errors::WhisperRealtimeError;

// TODO: nuke the legacy hand-rolled solution.
// TODO: test and benchmarks.

/// TODO: properly document this trait to explain why it's here, what it's for, and what's available
/// eg. Three provided VAD implementations: Silero, WebRtc, Earshot (a WebRtc) impl
/// tl;dr, Silero is very fast and efficient. Earshot should mainly be used as a fallback;
/// as far as I know, it's much less accurate, but it is very fast--YMMV.
pub trait VAD<T> {
    fn voice_detected(&mut self, samples: &[T]) -> bool;
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
    // but this function returns None if the Silero Vad struct fails to build.
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
                // TODO: this is not the right type of error; write a proper error to encapsulate the kind of error this is.
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
    pub fn try_new_whisper_default() -> Result<Self, WhisperRealtimeError> {
        SileroBuilder::new()
            .with_sample_rate(constants::WHISPER_SAMPLE_RATE as i64)
            .with_chunk_size(constants::SILERO_CHUNK_SIZE)
            .with_detection_probability_threshold(constants::VOICE_PROBABILITY_THRESHOLD)
            .build()
    }
}

impl<T: voice_activity_detector::Sample> VAD<T> for Silero {
    /// NOTE: This implementation assumes that the samples are at the same sample rate as the configured VAD
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        self.detection_probability_threshold > self.vad.predict(samples.iter().copied())
    }
    fn reset_session(&mut self) {
        // VoiceActivityDetector does not reset configurations to default settings when resetting the context
        // so this method is just a simple delegate.
        self.vad.reset()
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
            Self::R48kHz => 488000usize,
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
pub struct WebRTCBuilder {
    sample_rate: WebRtcSampleRate,
    aggressiveness: WebRtcFilterAggressiveness,
    frame_length: WebRtcFrameLengthMillis,
    detection_probability_threshold: f32,
}

impl WebRTCBuilder {
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
    pub fn build_webrtc_vad(self) -> Result<WebRtc, WhisperRealtimeError> {
        std::panic::catch_unwind(|| {
            webrtc_vad::Vad::new_with_rate_and_mode(
                self.sample_rate.to_webrtc_sample_rate(),
                self.aggressiveness.to_webrtc_vad_mode(),
            )
        })
        .map(|vad| WebRtc {
            vad,
            // WebRtcSampleRate and WebRtcFilterAgressiveness both have Copy semantics
            sample_rate: self.sample_rate,
            aggressiveness: self.aggressiveness,
            frame_length_in_ms: self.frame_length.to_ms(),
            detection_probability_threshold: self.detection_probability_threshold,
            // TODO: As above with the Silero impl, this requires a proper error member,
            // ParameterError is inappropriate and should be replaced
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
            frame_length_in_ms: 0,
            detection_probability_threshold: 0.0,
            prediction_predicate: predicate,
        })
    }
}

// TODO: properly document this to provide relevant information, like Silero.
pub struct WebRtc {
    vad: webrtc_vad::Vad,
    // Since webrtc_vad does not implement Copy or Clone, use the wrapper structs.
    // These have matching methods to produce the internal enumeration equivalent as needed
    sample_rate: WebRtcSampleRate,
    aggressiveness: WebRtcFilterAggressiveness,
    // This value will always be restricted to either 10ms, 20ms, 30ms as per the Enumeration
    // used in the accompanying builder
    frame_length_in_ms: usize,
    detection_probability_threshold: f32,
}

impl WebRtc {
    pub fn with_detection_probability_threshold(mut self, probability: f32) -> Self {
        self.detection_probability_threshold = probability;
        self
    }
    /// A "Default" whisper-ready WebRtc configuration
    pub fn try_new_whisper_default() -> Result<Self, WhisperRealtimeError> {
        WebRTCBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_detection_probability_threshold(constants::VOICE_PROBABILITY_THRESHOLD)
            .build_webrtc_vad()
    }
}

impl<T: IntoPcmS16 + Copy> VAD<T> for WebRtc {
    // NOTE: This implementation assumes that the samples are at the same sample rate as the configured VAD
    // Samples of insufficient length are zero-padded/truncated to avoid internal panicking.
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        let (int_audio, frame_size) = prepare_webrtc_frames(
            samples,
            self.frame_length_in_ms,
            self.sample_rate.to_sample_rate_hz(),
        );
        let frames = int_audio.chunks_exact(frame_size);

        let total_num_frames = frames.len();
        let speech_frames = frames.filter(|&frame| {
            // This should never, ever panic unless my arithmetic is busted
            // Unwrap to force a panic to catch errors in the implementation.
            self.vad
                .is_voice_segment(frame)
                .expect("The frame size should be valid.")
        });

        let voiced_proportion = (speech_frames.count() as f32) / (total_num_frames as f32);
        voiced_proportion > self.detection_probability_threshold
    }
    fn reset_session(&mut self) {
        // WebRtc_vad reverts to default settings when the context is cleared.
        // The vad must be reconfigured accordingly.
        self.vad.reset();
        self.vad
            .set_sample_rate(self.sample_rate.to_webrtc_sample_rate());
        self.vad.set_mode(self.aggressiveness.to_webrtc_vad_mode());
    }
}

type EarshotPredictionFilterPredicate =
    fn(&mut earshot::VoiceActivityDetector, &[i16]) -> Result<bool, earshot::Error>;
pub struct Earshot {
    vad: earshot::VoiceActivityDetector,
    // For dispatching the appropriate method
    sample_rate: usize,
    frame_length_in_ms: usize,
    detection_probability_threshold: f32,
    prediction_predicate: EarshotPredictionFilterPredicate,
}

impl Earshot {
    pub fn with_detection_probability_threshold(mut self, probability: f32) -> Self {
        self.detection_probability_threshold = probability;
        self
    }

    /// A "Default" whisper-ready Earhshot configuration
    pub fn try_new_whisper_default() -> Result<Self, WhisperRealtimeError> {
        WebRTCBuilder::new()
            .with_sample_rate(WebRtcSampleRate::R16kHz)
            .with_filter_aggressiveness(WebRtcFilterAggressiveness::LowBitrate)
            .with_detection_probability_threshold(constants::VOICE_PROBABILITY_THRESHOLD)
            .build_earshot()
    }
}

impl<T: IntoPcmS16 + Copy> VAD<T> for Earshot {
    fn voice_detected(&mut self, samples: &[T]) -> bool {
        let (int_audio, frame_size) =
            prepare_webrtc_frames(samples, self.frame_length_in_ms, self.sample_rate);
        let frames = int_audio.chunks_exact(frame_size);

        let total_num_frames = frames.len();
        let vad = &mut self.vad;
        let speech_frames = frames.filter(|&frame| {
            (self.prediction_predicate)(vad, frame).expect("Frame size should be valid.")
        });
        let voiced_proportion = (speech_frames.count() as f32) / (total_num_frames as f32);
        voiced_proportion > self.detection_probability_threshold
    }

    fn reset_session(&mut self) {
        // Earshot does not revert back to default settings when resetting the context,
        // so this method is just a delegate.
        self.vad.reset()
    }
}

/// This small utility function handles the padding/truncation required by WebRTC implementations
/// It returns the converted and padded audio, as well as the frame size so that methods consuming
/// this function do not need to recompute the frame size.
fn prepare_webrtc_frames<T: IntoPcmS16 + Copy>(
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

// LEGACY

// This is for the naive strategy to avoid extra memory allocations at runtime.
static FFT_PLANNER: LazyLock<Mutex<RealFftPlanner<f64>>> =
    LazyLock::new(|| Mutex::new(RealFftPlanner::<f64>::new()));

// Silero is considerably faster than my hand-rolled VAD. Only use the hand-rolled VAD if silero is
// somehow not supported.
// If using the enum, the discriminant should contain the object required to compute the voice activity probability.
pub enum VadStrategy {
    Naive,
    Silero,
}

impl Default for VadStrategy {
    fn default() -> Self {
        Self::Silero
    }
}

// This doesn't really make sense together as a trait.
// TODO: split into structs SileroVoiceActivityDetection, and NaiveVoiceActivityDetection or some flavour thereof
// and implement is_voice_detected(samples)
// Take configurations in as an object or as parameters
pub trait VoiceActivityDetection<
    T: voice_activity_detector::Sample
        + num_traits::cast::NumCast
        + num_traits::cast::FromPrimitive
        + num_traits::cast::ToPrimitive
        + num_traits::Zero,
>
{
    fn is_voice_detected_silero(
        vad: &mut voice_activity_detector::VoiceActivityDetector,
        samples: &Vec<T>,
        voice_probability_threshold: f32,
    ) -> bool {
        let s = samples.clone();
        let probability = vad.predict(s);
        probability > voice_probability_threshold
    }

    fn is_voice_detected_naive(
        sample_rate: f64,
        samples: &Vec<T>,
        voice_energy_threshold: f64,
        window_len: f64,
        window_step: f64,
        freq_threshold: f64,
        voice_probability_threshold: f32,
    ) -> Result<bool, WhisperRealtimeError> {
        let mut samples_f64: Vec<f64> = samples
            .iter()
            .map(|n| n.to_f64().expect("Failed to convert T to f64"))
            .collect();

        if freq_threshold > 0.0f64 {
            let original_rms = calculate_rms(&samples_f64);
            high_pass_filter(&mut samples_f64, freq_threshold, sample_rate);
            let filtered_rms = calculate_rms(&samples_f64);
            let gain = original_rms / filtered_rms;
            apply_gain(&mut samples_f64, gain);
        }

        // run a DFT and use short time energy based VAD.
        let dft_vad = naive_frame_energy_vad(
            &samples_f64,
            sample_rate,
            voice_energy_threshold,
            window_len,
            window_step,
            constants::E0,
        );

        if let Err(e) = dft_vad {
            return Err(e);
        }

        let vad = dft_vad.unwrap().0;

        let mean: f32 = vad.iter().fold(0.0, |acc, n| acc + *n as f32) / vad.len() as f32;
        Ok(mean > voice_probability_threshold)
    }
}

// This mutates the samples to remove low frequency sounds.
fn high_pass_filter(samples: &mut Vec<f64>, frequency_threshold: f64, sample_rate: f64) {
    let rc: f64 = 1.0 / (2.0 * PI * frequency_threshold);
    let dt: f64 = 1.0 / sample_rate;

    let alpha: f64 = dt / (rc + dt);

    let mut y = samples[0];

    for i in 1..samples.len() {
        y = alpha * (y + samples[i] - samples[i - 1]);

        samples[i] = y;
    }
}

// TODO: if keeping this implementation, expose the functions.
// Audio volume correction after high-pass
fn calculate_rms(data: &[f64]) -> f64 {
    let sum_of_squares: f64 = data.iter().fold(0.0, |acc, x| acc + (*x).powi(2));
    (sum_of_squares / data.len() as f64).sqrt()
}

fn apply_gain(data: &mut [f64], gain: f64) {
    for x in data.iter_mut() {
        *x *= gain;
    }
}

// The naive VAD is a port of the code used in this blog post by Ayoub Malek:
// https://superkogito.github.io/blog/2020/02/09/naive_vad.html
//
// I do not have any knowledge in this domain yet, nor do I have much knowledge of
// python/numpy.

// At this time, generics are not needed & would complicate the implementation.

/// A basic  Median filter
fn medfilt(data: &[f64], kernel_size: usize) -> Vec<f64> {
    let mut filtered = vec![0.0; data.len()];
    let k = kernel_size / 2;
    for i in 0..data.len() {
        let start = if i < k { 0 } else { i - k };
        let end = if i + k >= data.len() {
            data.len() - 1
        } else {
            i + k
        };
        let mut window: Vec<f64> = data[start..=end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        filtered[i] = window[window.len() / 2];
    }
    filtered
}

fn repeat_elements(data: &[f64], repeat: usize) -> Vec<f64> {
    data.iter()
        .flat_map(|&x| std::iter::repeat(x).take(repeat))
        .collect()
}

//
fn stride_trick(array: &[f64], stride_length: usize, stride_step: usize) -> Vec<Vec<f64>> {
    let n = ((array.len() - stride_length) / stride_step) + 1;

    (0..n)
        .map(|i| array[i * stride_step..i * stride_step + stride_length].to_vec())
        .collect()
}

fn framing(
    samples: &[f64],
    sample_rate: f64,
    window_len: f64,
    window_step: f64,
) -> Result<(Vec<Vec<f64>>, usize), WhisperRealtimeError> {
    if window_len < window_step {
        return Err(WhisperRealtimeError::ParameterError(format!(
            "Framing: window_len, {} must be larger than window_hop, {}",
            window_len, window_step
        )));
    }

    let frame_length = (window_len * sample_rate) as usize;
    let frame_step = (window_step * sample_rate) as usize;

    let signal_length = samples.len();
    let frames_overlap = frame_length - frame_step;

    let rest_samples =
        (signal_length.abs_diff(frames_overlap)) % (frame_length.abs_diff(frames_overlap));

    let pad_samples: Vec<f64> = samples
        .iter()
        .cloned()
        .chain(vec![0.0; frame_step - rest_samples].into_iter())
        .collect();

    let frames = stride_trick(&pad_samples, frame_length, frame_step);
    Ok((frames, frame_length))
}

fn calculate_normalized_short_time_energy(
    frames: &[Vec<f64>],
) -> Result<Vec<f64>, WhisperRealtimeError> {
    if frames.len() < 1 {
        return Err(WhisperRealtimeError::ParameterError(
            "STE: cannot calculate from 0-length array".to_owned(),
        ));
    }

    let planner = &mut FFT_PLANNER.lock().expect("Failed to get FFT mutex");

    let fft = planner.plan_fft_forward(frames[0].len());

    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    Ok(frames
        .iter()
        .map(|frame| {
            input.copy_from_slice(frame);
            fft.process(&mut input, &mut output)
                .expect("Failed to process fft");

            output.iter().map(|c| c.norm_sqr()).sum::<f64>() / (frame.len() as f64).powi(2)
        })
        .collect())
}

// Array of 0/1 and the vector of voice frames.
fn naive_frame_energy_vad(
    samples: &[f64],
    sample_rate: f64,
    threshold: f64,
    window_len: f64,
    window_hop: f64,
    e0: f64,
) -> Result<(Vec<u8>, Vec<f64>), WhisperRealtimeError> {
    let result = framing(samples, sample_rate, window_len, window_hop);

    if let Err(e) = result {
        return Err(e);
    }

    let (frames, frames_len) = result.unwrap();

    // Compute STE -> Voiced frames
    let normalized_energy = calculate_normalized_short_time_energy(&frames);
    if let Err(e) = normalized_energy {
        return Err(e);
    }
    let energy = normalized_energy.unwrap();

    let log_energy: Vec<f64> = energy.iter().map(|e| 10.0 * (*e / e0).log10()).collect();

    let filtered_energy = medfilt(&log_energy, constants::KERNEL_SIZE);
    let repeated_energy = repeat_elements(&filtered_energy, frames_len);

    let mut vad: Vec<u8> = vec![0; repeated_energy.len()];

    let f_frames: Vec<f64> = frames.iter().flatten().map(|n| *n).collect();

    assert_eq!(
        vad.len(),
        f_frames.len(),
        "Frame padding is wrong. Vad Len: {} F_Frame Len: {}",
        &vad.len(),
        &f_frames.len()
    );

    let mut v_frames: Vec<f64> = Vec::new();
    for (i, &e) in repeated_energy.iter().enumerate() {
        if e > threshold {
            vad[i] = 1;
            v_frames.push(f_frames[i]);
        }
    }

    Ok((vad, v_frames))
}

// IMPLEMENTATION TESTS
// TODO: this should in some way exist in the testing folder.
// noinspection DuplicatedCode
#[cfg(test)]
mod vad_tests {
    use hound;
    use hound::SampleFormat;
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;

    use crate::whisper::configs;

    use super::*;

    pub struct AudioTester;

    impl VoiceActivityDetection<f32> for AudioTester {}

    #[test]
    #[ignore]
    fn test_framing() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let configs = configs::WhisperConfigsV1::default();
        let audio: Vec<f64> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f64
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized
                })
                .collect(),
        };

        let result = framing(
            &audio,
            sample_rate as f64,
            configs.naive_window_len,
            configs.naive_window_step,
        );
        assert!(result.is_ok(), "{}", result.err().unwrap());

        let audio: Vec<f64> = result.unwrap().0.iter().flatten().map(|n| *n).collect();
        // Write the output in f32, regular spec.

        let write_spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut writer =
            hound::WavWriter::create("tests/audio_files/test_flatten_f32.wav", write_spec)
                .expect("Failed to create wav writer");

        for i in 0..audio.len() {
            writer
                .write_sample(audio[i] as f32)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize write");

        // De-normalize back to i16 & write
        let write_spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer =
            hound::WavWriter::create("tests/audio_files/test_flatten_i16.wav", write_spec)
                .expect("Failed to create wav writer");

        for i in 0..audio.len() {
            let sample = audio[i];
            let normalized = if sample.is_sign_positive() {
                (audio[i] * (i16::MAX as f64)) as i16
            } else {
                (-1f64 * audio[i] * (i16::MIN as f64)) as i16
            };

            writer
                .write_sample(normalized)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize write");
    }

    #[test]
    #[ignore]
    fn test_high_pass_output() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let configs = configs::WhisperConfigsV1::default();
        let mut audio: Vec<f64> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f64
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized
                })
                .collect(),
        };

        let original_rms = calculate_rms(&audio);

        high_pass_filter(
            &mut audio,
            configs.naive_vad_freq_threshold,
            sample_rate as f64,
        );

        let filtered_rms = calculate_rms(&audio);
        let gain = original_rms / filtered_rms;

        apply_gain(&mut audio, gain);

        let write_spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut writer =
            hound::WavWriter::create("tests/audio_files/test_high_pass.wav", write_spec)
                .expect("Failed to create wav writer");

        for i in 0..audio.len() {
            writer
                .write_sample(audio[i] as f32)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize write");
    }

    #[test]
    #[ignore]
    fn test_naive_frame_energy_vad() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let audio: Vec<f64> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f64
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized
                })
                .collect(),
        };

        let mut configs = configs::WhisperConfigsV1::default();
        // Given the length of the audio track, this is being reduced to avoid
        // unnecessary panicking.
        // The provided audio track has around 60% speech.
        configs.voice_probability_threshold = 0.5;

        let result = naive_frame_energy_vad(
            &audio,
            sample_rate as f64,
            configs.naive_vad_energy_threshold,
            configs.naive_window_len,
            configs.naive_window_step,
            constants::E0,
        );

        assert!(result.is_ok(), "{}", result.err().unwrap());

        let (vad, v_frames) = result.unwrap();

        let mean: f32 = vad.iter().fold(0.0, |acc, n| acc + *n as f32) / vad.len() as f32;
        assert!(
            mean > configs.voice_probability_threshold,
            "Voice not properly detected. Computed mean: {} > Threshold: {}",
            mean,
            configs.voice_probability_threshold
        );

        assert!(v_frames.len() > 0, "No vframes detected");

        // Test writing - not working.
        let write_spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer =
            hound::WavWriter::create("tests/audio_files/test_naive_vframes_f32.wav", write_spec)
                .expect("Failed to create wav writer");

        for i in 0..v_frames.len() {
            writer
                .write_sample(v_frames[i] as f32)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize write");
    }
    #[test]
    #[ignore]
    fn test_naive_voice_detection() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let audio: Vec<f32> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized as f32
                })
                .collect(),
        };

        let mut configs = configs::WhisperConfigsV1::default();
        // No high-pass
        configs.naive_vad_freq_threshold = 0.0;
        // The audio track contains around 60% speech.
        configs.voice_probability_threshold = 0.5;
        let result = AudioTester::is_voice_detected_naive(
            sample_rate as f64,
            &audio,
            configs.naive_vad_energy_threshold,
            configs.naive_window_len,
            configs.naive_window_step,
            configs.naive_vad_freq_threshold,
            configs.voice_probability_threshold,
        );

        assert!(result.is_ok(), "{}", result.err().unwrap());

        let voice_detected = result.unwrap();
        assert!(voice_detected, "Failed to detect voice");
    }

    #[test]
    #[ignore]
    fn test_naive_voice_detection_with_high_pass() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let audio: Vec<f32> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized as f32
                })
                .collect(),
        };

        let mut configs = configs::WhisperConfigsV1::default();

        // The audio track contains around 60% speech.
        configs.voice_probability_threshold = 0.5;
        let result = AudioTester::is_voice_detected_naive(
            sample_rate as f64,
            &audio,
            configs.naive_vad_energy_threshold,
            configs.naive_window_len,
            configs.naive_window_step,
            configs.naive_vad_freq_threshold,
            configs.voice_probability_threshold,
        );

        assert!(result.is_ok(), "{}", result.err().unwrap());

        let voice_detected = result.unwrap();
        assert!(voice_detected, "Failed to detect voice");
    }

    #[test]
    #[ignore]
    fn test_silero_vad() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let audio: Vec<f32> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized as f32
                })
                .collect(),
        };

        // This might not be mutable.
        let mut configs = configs::WhisperConfigsV1::default();

        let mut vad = voice_activity_detector::VoiceActivityDetector::builder()
            .sample_rate(sample_rate as i64)
            .chunk_size(1024usize)
            .build()
            .expect("Failed to build voice activity detector");

        // Break into 10s chunks.
        let secs = 10;
        let audio_chunks = audio.chunks_exact(sample_rate as usize * secs);
        let len = audio_chunks.len();
        let sum: u64 = audio_chunks.fold(0, |acc, v| {
            if AudioTester::is_voice_detected_silero(
                &mut vad,
                &v.to_vec(),
                configs.voice_probability_threshold,
            ) {
                acc + 1
            } else {
                acc
            }
        });

        // Reducing the threshold a little bit
        // The audio track contains around 60% speech.
        configs.voice_probability_threshold = 0.5;

        let mean = sum as f64 / len as f64;
        assert!(
            mean > configs.voice_probability_threshold as f64,
            "Failed to detect voice. Mean: {}",
            mean
        );
    }

    // TODO: this should be moved to benchmarks.
    // Also, Silero is considerably faster than the hand-rolled solution.
    // TODO: consider removing the Naive VAD implementation.
    #[test]
    #[ignore]
    fn speed_test() {
        let mut time = std::time::Instant::now();
        (0..1000)
            .into_par_iter()
            .for_each(|_| test_naive_voice_detection());
        let mut now = std::time::Instant::now();
        let naive_diff = (now - time).as_millis();

        time = std::time::Instant::now();

        (0..1000)
            .into_par_iter()
            .for_each(|_| test_naive_voice_detection_with_high_pass());

        now = std::time::Instant::now();

        let high_pass_diff = (now - time).as_millis();

        time = std::time::Instant::now();

        (0..1000).into_par_iter().for_each(|_| test_silero_vad());

        now = std::time::Instant::now();
        let silero_diff = (now - time).as_millis();

        // Need to use --nocapture or --show-output to get this output.
        println!(
            "Per 1000 iterations: \n\
        Naive: {}ms\n\
        High-Pass: {}ms\n\
        Silero: {}ms",
            naive_diff, high_pass_diff, silero_diff
        );
    }
}
