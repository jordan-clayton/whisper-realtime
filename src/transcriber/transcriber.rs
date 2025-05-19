use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::utils::callback::Callback;
use crate::utils::errors::WhisperRealtimeError;

pub trait WhisperProgressCallback: Callback<Argument = i32> + Send + Sync + 'static {}
impl<T: Callback<Argument = i32> + Send + Sync + 'static> WhisperProgressCallback for T {}

pub trait Transcriber {
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, WhisperRealtimeError>;
}

pub trait CallbackTranscriber<P>: Transcriber
where
    P: WhisperProgressCallback,
{
    fn process_with_callbacks(
        &mut self,
        run_transcription: Arc<AtomicBool>,
        callbacks: WhisperCallbacks<P>,
    ) -> Result<String, WhisperRealtimeError>;
}

/// This struct encapsulates various whisper callbacks which can be set before running transcription
/// Other callbacks will be added as needed/desired. File an issue to bring attention to this.
/// At the moment, all fields are public; encapsulation is not needed at this time.
pub struct WhisperCallbacks<P>
where
    P: WhisperProgressCallback,
{
    pub progress: Option<P>,
}

impl<P> WhisperCallbacks<P> where P: WhisperProgressCallback {}

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
