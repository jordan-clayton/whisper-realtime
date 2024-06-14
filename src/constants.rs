const KEEP_MS: f64 = 200f64;
const SAMPLE_RATE: f64 = 16000f64;

pub const N_SAMPLES_KEEP: usize = ((1e-3 * KEEP_MS) * SAMPLE_RATE) as usize;

pub const N_SAMPLES_30S: usize = ((1e-3 * 30000.0) * SAMPLE_RATE) as usize;
pub const INPUT_BUFFER_CAPACITY: usize = N_SAMPLES_30S;
pub const AUDIO_BUFFER_CAPACITY: usize = N_SAMPLES_30S;

// TODO: figure out an output buffer size.

pub const OUTPUT_BUFFER_CAPACITY: usize = N_SAMPLES_30S;

pub const MAX_QUEUE_ERRORS: usize = 5;

// This is 120 bytes -> Consider refactoring.
pub const CPAL_ERROR_BUFFER_CAPACITY: usize =
    std::mem::size_of::<cpal::StreamError>() * MAX_QUEUE_ERRORS;

// Not sure how big to make this.
pub const ERROR_BUFFER_CAPACITY: usize = 2048;
