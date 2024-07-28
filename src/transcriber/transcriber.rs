pub trait Transcriber {
    fn process_audio(&mut self, whisper_state: &mut whisper_rs::WhisperState) -> String;
    // fn convert_input_audio(input: &[u8], sample_format: cpal::SampleFormat) -> Vec<f32>;
    // fn convert_to_i16_sample(byte_chunks: usize, buffer: &Vec<u8>) -> Vec<f32>;
    fn set_full_params<'a>(
        full_params: &mut whisper_rs::FullParams<'a, 'a>,
        prefs: &'a crate::configs::Configs,
        tokens: Option<&'a Vec<std::ffi::c_int>>,
    ) {
        let lang = prefs.set_language.as_ref();

        full_params.set_n_threads(prefs.n_threads);
        full_params.set_n_threads(prefs.n_threads);
        full_params.set_translate(prefs.set_translate);

        if let Some(language) = lang {
            full_params.set_language(Some(language.as_str()));
        } else {
            full_params.set_language(Some("auto"));
        }

        // // Stdio only
        full_params.set_print_special(prefs.print_special);
        full_params.set_print_progress(prefs.print_progress);
        full_params.set_print_realtime(prefs.print_realtime);
        full_params.set_print_timestamps(prefs.print_timestamps);

        if let Some(token_buffer) = tokens {
            full_params.set_tokens(&token_buffer);
        }
    }
}
