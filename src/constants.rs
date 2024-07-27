const KEEP_MS: f64 = 200f64;
pub const WHISPER_SAMPLE_RATE: f64 = 16000f64;

pub const N_SAMPLES_KEEP: usize = ((1e-3 * KEEP_MS) * WHISPER_SAMPLE_RATE) as usize;

pub const N_SAMPLES_30S: usize = ((1e-3 * 30000.0) * WHISPER_SAMPLE_RATE) as usize;

pub const N_SAMPLES_2S: usize = ((1e-3 * 2000.0) * WHISPER_SAMPLE_RATE) as usize;

// Total length is 10s
pub const INPUT_BUFFER_CAPACITY: usize = 10000;

// This is currently set at 1 hr -> This has not yet been tested for OOM panics.
pub const REALTIME_AUDIO_TIMEOUT: u128 = std::time::Duration::new(3600 * 60, 0).as_millis();

// in ms
pub const AUDIO_SAMPLE_MS: usize = 10000;
pub const VAD_SAMPLE_MS: usize = 300;
pub const PHRASE_TIMEOUT: usize = 3000;
pub const PAUSE_DURATION: u64 = 100;

// This works best when within the range of 0.65-0.80
// Higher thresholds are prone to false negatives.
pub const VOICE_THRESHOLD: f32 = 0.65;

// This is currently only used for testing purposes.
pub const SAMPLE_DURATION: usize = 10000;
