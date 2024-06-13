// TODO: Downloading trait?
// TODO: File load trait?
// TODO: Deleting models trait?
// ^^ These have implementations - generalize

// -> These new functions probably need to be handled separately
// for realtime/static.
pub trait Transcriber<'a: 'b, 'b> {
    fn process_audio(&'a mut self) -> String;
    // TODO: this could be moved to a transcriber impl
    fn convert_input_audio(input: &[u8], sample_format: cpal::SampleFormat) -> Vec<f32>;
    // TODO: this could be moved to a transcriber impl
    fn convert_to_i16_sample(byte_chunks: usize, buffer: &Vec<u8>) -> Vec<f32>;
    // TODO: this could be moved to a transcriber impl
    fn set_full_params<'b, 'a: 'b>(
        full_params: &mut whisper_rs::FullParams<'b, 'b>,
        prefs: &'a crate::preferences::Configs,
        tokens: Option<&'b Vec<std::ffi::c_int>>,
    );
}

pub trait Queue<T> {
    fn new() -> Self;
    fn with_capacity(size: usize) -> Self;

    fn resize(&mut self, new_len: usize);
    fn resize_with(&mut self, new_len: usize, generator: impl FnMut() -> T);
    fn pop(&mut self) -> T;
    fn push(&mut self, item: &T);

    fn peek(&self) -> T;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn is_empty(&self) -> bool;
}
