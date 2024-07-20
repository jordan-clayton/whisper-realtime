const KEEP_MS: f64 = 200f64;
pub const SAMPLE_RATE: f64 = 16000f64;

pub const N_SAMPLES_KEEP: usize = ((1e-3 * KEEP_MS) * SAMPLE_RATE) as usize;

pub const N_SAMPLES_30S: usize = ((1e-3 * 30000.0) * SAMPLE_RATE) as usize;

pub const N_SAMPLES_2S: usize = ((1e-3 * 2000.0) * SAMPLE_RATE) as usize;

// Total length is 10s
pub const INPUT_BUFFER_CAPACITY: usize = 10000;
// This is currently set at 10 mins.
pub const REALTIME_AUDIO_TIMEOUT: u128 = std::time::Duration::new(10 * 60, 0).as_millis();
// TODO: Some of these should be parameters.
pub const AUDIO_CHUNK_SIZE: usize = 10000;
pub const VAD_CHUNK_SIZE: usize = 300;
pub const PHRASE_TIMEOUT: usize = 3000;
pub const VOICE_THRESHOLD: f32 = 0.65;
pub const PAUSE_DURATION: u64 = 100;
pub const SAMPLE_DURATION: usize = 10000;
