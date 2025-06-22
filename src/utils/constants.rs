const KEEP_MS: f64 = 300f64;
pub const WHISPER_SAMPLE_RATE: f64 = 16000f64;
pub const SILERO_CHUNK_SIZE: usize = 512;

// To help with resolving word boundaries.
pub const N_SAMPLES_KEEP: usize = ((1e-3 * KEEP_MS) * WHISPER_SAMPLE_RATE) as usize;

pub const N_SAMPLES_30S: usize = ((1e-3 * 30000.0) * WHISPER_SAMPLE_RATE) as usize;

pub const N_SAMPLES_2S: usize = ((1e-3 * 2000.0) * WHISPER_SAMPLE_RATE) as usize;
pub const N_SEGMENTS_DIFF: usize = 3;
pub const WORKING_SET_SIZE: usize = N_SEGMENTS_DIFF * 5;

// Conservatively at 90% match; might need to be shorter/exposed as a parameter
// Let's test this at a very low threshold to see as to what's going on
pub const DIFF_THRESHOLD_HIGH: f64 = 0.9;
pub const DIFF_THRESHOLD_MED: f64 = 0.7;
pub const DIFF_THRESHOLD_LOW: f64 = 0.50;
pub const DIFF_THRESHOLD_MIN: f64 = 0.40;

// Conservatively put this at 2 seconds might need to be shorter/exposed as a parameter.
// It might work better if things are extremely strict
pub const TIMESTAMP_GAP: i64 = 3000;
pub const TIMESTAMP_EPSILON: i64 = 10;

// Total length is 10s
pub const INPUT_BUFFER_CAPACITY: usize = 10000;

// This is currently set at 1 hr -> This has not yet been tested for OOM panics.
pub const REALTIME_AUDIO_TIMEOUT: u128 = std::time::Duration::new(3600, 0).as_millis();

pub const AUDIO_BUFFER_SIZE: u16 = 1024;

pub const BLANK_AUDIO: &str = "[BLANK_AUDIO]";
pub const ELLIPSIS: &str = "...";

// in ms
pub const AUDIO_SAMPLE_MS: usize = 10000;
pub const VAD_SAMPLE_MS: usize = 300;
pub const PAUSE_DURATION: u64 = 100;
pub const MAX_PROMPT_TOKENS: usize = 16384;

// THESE ARE RECOMMENDATIONS
// This works best when within the range of 0.65-0.80
// Higher thresholds are prone to false negatives.
pub const SILERO_VOICE_PROBABILITY_THRESHOLD: f32 = 0.65;
pub const WEBRTC_VOICE_PROBABILITY_THRESHOLD: f32 = 0.70;
pub const OFFLINE_VOICE_PROBABILITY_THRESHOLD: f32 = 0.75;

// This is currently only used for testing purposes.
pub const SAMPLE_DURATION: usize = 10000;

pub const CONVERT_STEREO_TO_MONO: u8 = 0b01;
pub const CONVERT_MONO_TO_STEREO: u8 = 0b10;
