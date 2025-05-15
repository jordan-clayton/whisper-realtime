use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::whisper::configs::WhisperConfigsV1;

// TODO: rethink arguments, it should be possible to encapsulate the whisper state/context within the transcriber object
// There's no reason as to why the context cannot be constructed/owned by the transcriber
pub trait Transcriber {
    fn process_audio(
        &mut self,
        whisper_state: &mut whisper_rs::WhisperState,
        run_transcription: Arc<AtomicBool>,
        progress_callback: Option<impl FnMut(i32) + Send + Sync + 'static>,
    ) -> String;
    // TODO: Rethink this. Setting full_params should only really need to happen once.
    // Also: It doesn't necessarily need to be a trait method.
    // Also twice: It makes more sense as a function that returns Whisper FullParams, or perhaps a builder extension
    fn set_full_params<'a>(
        full_params: &mut whisper_rs::FullParams<'a, 'a>,
        prefs: &'a Arc<WhisperConfigsV1>,
        tokens: Option<&'a Vec<std::ffi::c_int>>,
    ) {
        full_params.set_n_threads(prefs.n_threads);
        full_params.set_n_threads(prefs.n_threads);
        full_params.set_translate(prefs.set_translate);
        // If set to translation mode, auto-detection is required.
        full_params.set_detect_language(prefs.set_translate);

        if let Some(language) = &prefs.language {
            full_params.set_language(Some(language.as_str()));
        } else {
            full_params.set_language(None)
        }

        // Stdio only
        full_params.set_print_special(prefs.print_special);
        full_params.set_print_progress(prefs.print_progress);
        full_params.set_print_realtime(prefs.print_realtime);
        full_params.set_print_timestamps(prefs.print_timestamps);

        if let Some(token_buffer) = tokens {
            full_params.set_tokens(&token_buffer);
        }
    }
}
