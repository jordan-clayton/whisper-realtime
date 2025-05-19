use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::utils::callback::Callback;
use crate::utils::errors::WhisperRealtimeError;

pub trait Transcriber {
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, WhisperRealtimeError>;
}

pub enum WhisperCallback<CB: Callback<Argument = i32> + Send + Sync> {
    Loadable(CB),
    Loaded,
}

impl<CB: Callback<Argument = i32> + Send + Sync> WhisperCallback<CB> {
    pub fn try_get_progress_callback(&mut self) -> Option<CB> {
        match std::mem::replace(self, WhisperCallback::Loaded) {
            WhisperCallback::Loadable(callback) => Some(callback),
            WhisperCallback::Loaded => None,
        }
    }

    /// If and only if the callback is the exact same type, can it be loaded into the old
    /// callback object. The callback must also not already have a callback.
    pub fn return_callback(&mut self, cb: CB) -> Result<(), WhisperRealtimeError> {
        if let WhisperCallback::Loadable(_) = self {
            return Err(WhisperRealtimeError::ParameterError(
                "Whisper Callback already loaded.".to_string(),
            ));
        }
        *self = WhisperCallback::Loadable(cb);
        Ok(())
    }
}

pub enum WhisperOutput {
    ContinuedPhrase(String),
    FinishedPhrase(String),
}

impl WhisperOutput {
    pub fn inner(&self) -> &String {
        match self {
            WhisperOutput::ContinuedPhrase(msg) | WhisperOutput::FinishedPhrase(msg) => msg,
        }
    }
    pub fn into_inner(self) -> String {
        match self {
            WhisperOutput::ContinuedPhrase(msg) | WhisperOutput::FinishedPhrase(msg) => msg,
        }
    }
}
