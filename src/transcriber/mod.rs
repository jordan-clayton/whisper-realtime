use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use strum::{Display, EnumString, IntoStaticStr};

use crate::utils::callback::Callback;
use crate::utils::errors::WhisperRealtimeError;

pub mod offline_transcriber;
pub mod realtime_transcriber;
pub mod vad;

// TODO: convenience functions for just one-and-done running offline/realtime transcription
// TODO: finish documenting
pub trait OfflineWhisperProgressCallback: Callback<Argument = i32> + Send + Sync + 'static {}
impl<T: Callback<Argument = i32> + Send + Sync + 'static> OfflineWhisperProgressCallback for T {}

pub trait Transcriber {
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, WhisperRealtimeError>;
}

pub trait CallbackTranscriber<P>: Transcriber
where
    P: OfflineWhisperProgressCallback,
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
    P: OfflineWhisperProgressCallback,
{
    pub progress: Option<P>,
}

impl<P> WhisperCallbacks<P> where P: OfflineWhisperProgressCallback {}

#[derive(Clone)]
pub struct WhisperSegment {
    pub text: String,
    // These are measured in milliseconds.
    // Whisper.cpp stores its timestamps in units of 0.1 ms
    pub start_time: i64,
    pub end_time: i64,
}

/// WhisperOutput: Enum for different types of output sendable through a Transcriber channel
/// ConfirmedTranscription: contains the known and confidently transcribed output thus far
/// CurrentSegments: contains a collection of the working set of segments. These get confirmed when VAD detects no voice.
/// ControlPhrase: WhisperControlPhrase, an enum to indicate the state of the transcription computation; suggested to use for UI
pub enum WhisperOutput {
    ConfirmedTranscription(String),
    CurrentSegments(Vec<String>),
    // TODO: swap this to another Enumeration so that it can be matched for control flow
    ControlPhrase(WhisperControlPhrase),
}

impl WhisperOutput {
    pub fn into_inner(self) -> String {
        match self {
            WhisperOutput::ConfirmedTranscription(msg) => msg,
            WhisperOutput::CurrentSegments(segments) => segments.join(""),
            WhisperOutput::ControlPhrase(control_phrase) => control_phrase.to_string(),
        }
    }
}

// These would benefit from some eventual localization
#[derive(EnumString, IntoStaticStr, Display)]
pub enum WhisperControlPhrase {
    #[strum(serialize = "[GETTING_READY]")]
    GettingReady,
    #[strum(serialize = "[START SPEAKING]")]
    StartSpeaking,
    #[strum(serialize = "[TRANSCRIPTION TIMEOUT]")]
    TranscriptionTimeout,
    #[strum(serialize = "[END TRANSCRIPTION]")]
    EndTranscription,
    #[strum(serialize = "Debug: {0}")]
    Debug(String),
}
