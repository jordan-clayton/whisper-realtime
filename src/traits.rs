// TODO: Downloading trait?
//
//
// NOTE: MIGRATE BELOW TO GUI.
// TODO: File load trait?
// TODO: Deleting models trait?


// TODO: migrate to transcriber mod.
pub trait Transcriber {
    fn process_audio(&mut self, whisper_state: &mut whisper_rs::WhisperState) -> String;
    // fn convert_input_audio(input: &[u8], sample_format: cpal::SampleFormat) -> Vec<f32>;
    // fn convert_to_i16_sample(byte_chunks: usize, buffer: &Vec<u8>) -> Vec<f32>;
    fn set_full_params<'a>(
        full_params: &mut whisper_rs::FullParams<'a, 'a>,
        prefs: &'a crate::configs::Configs,
        tokens: Option<&'a Vec<std::ffi::c_int>>,
    );
}

pub trait Downloader {

}

// TODO: remove, is currently unusued.
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
