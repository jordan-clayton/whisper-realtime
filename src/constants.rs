const KEEP_MS: f64 = 200f64;
pub const SAMPLE_RATE: f64 = 16000f64;

pub const N_SAMPLES_KEEP: usize = ((1e-3 * KEEP_MS) * SAMPLE_RATE) as usize;

pub const N_SAMPLES_30S: usize = ((1e-3 * 30000.0) * SAMPLE_RATE) as usize;

pub const N_SAMPLES_2S: usize = ((1e-3 * 2000.0) * SAMPLE_RATE) as usize;

// Total length is 10s
pub const INPUT_BUFFER_CAPACITY: usize = 10000;

// TODO: Figure out a microphone buffer size

pub const AUDIO_CHUNK_SIZE: usize = 5000;

pub const VOICE_THRESHOLD: f32 = 0.75;

pub const PAUSE_DURATION: u64 = 100;

pub const TEST_DURATION: usize = 2000;
pub const SAMPLE_DURATION: usize = 10000;
pub const AUDIO_BUFFER_CAPACITY: usize = N_SAMPLES_30S * std::mem::size_of::<f32>();

// TODO: figure out an output buffer size.
// 2048 characters?
pub const OUTPUT_BUFFER_CAPACITY: usize = 2048 * std::mem::size_of::<char>();

pub const MAX_QUEUE_ERRORS: usize = 5;

// Not sure how big to make this.
pub const ERROR_BUFFER_CAPACITY: usize = 2048;
