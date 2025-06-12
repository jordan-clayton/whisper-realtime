use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use strum::{Display, EnumString, IntoStaticStr};

use crate::utils::callback::Callback;
use crate::utils::errors::RibbleWhisperError;

pub mod offline_transcriber;
pub mod realtime_transcriber;
pub mod vad;

// Trait alias, used until the feature reaches stable
pub trait OfflineWhisperProgressCallback: Callback<Argument = i32> + Send + Sync + 'static {}
impl<T: Callback<Argument = i32> + Send + Sync + 'static> OfflineWhisperProgressCallback for T {}

#[inline]
pub fn redirect_whisper_logging_to_hooks() {
    whisper_rs::install_logging_hooks()
}

/// Handles running Whisper transcription
pub trait Transcriber {
    /// Loads a compatible whisper model, sets up the whisper state and runs the full model
    /// # Arguments
    /// * run_transcription: Arc<AtomicBool>, a shared flag used to indicate when to stop transcribing
    /// # Returns
    /// * Ok(String) on success, Err(WhisperRealtimeError) on failure
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, RibbleWhisperError>;
}

/// Handles running Whisper transcription, with support for optional callbacks
/// These callbacks are called from whisper so their safety cannot be completely guaranteed.
/// However, since these callbacks do not touch the whisper state, they should work as expected.
pub trait CallbackTranscriber<P>: Transcriber
where
    P: OfflineWhisperProgressCallback,
{
    fn process_with_callbacks(
        &mut self,
        run_transcription: Arc<AtomicBool>,
        callbacks: WhisperCallbacks<P>,
    ) -> Result<String, RibbleWhisperError>;
}

/// Encapsulates various whisper callbacks which can be set before running transcription
/// Other callbacks will be added as needed.
pub struct WhisperCallbacks<P: OfflineWhisperProgressCallback> {
    /// Optional progress callback
    pub progress: Option<P>,
}

/// Encapsulates a whisper segment with start and end timestamps
#[derive(Clone)]
pub struct WhisperSegment {
    /// Segment text
    pub text: String,
    /// Timestamp start time, measured in 0.1 ms
    pub start_time: i64,
    /// Timestamp end time, measured in 0.1 ms
    pub end_time: i64,
}

/// Encapsulates possible types of output sent through a Transcriber channel
/// NOTE: Outputs with accompanying timestamps are not yet implemented.
pub enum WhisperOutput {
    /// Contains the most confident state of the transcription as a string
    ConfirmedTranscription(String),
    /// Contains a copy of the current working set of segments.
    /// When word-boundaries are confirmed to be resolved, they become part of the output
    /// and are removed from the working set
    CurrentSegments(Vec<String>),
    /// For sending running state and control messages from the Transcriber
    ControlPhrase(WhisperControlPhrase),
}

impl WhisperOutput {
    // Consumes and extracts the inner contents of a WhisperOUtput into a string
    pub fn into_inner(self) -> String {
        match self {
            WhisperOutput::ConfirmedTranscription(msg) => msg,
            WhisperOutput::CurrentSegments(segments) => segments.join(""),
            WhisperOutput::ControlPhrase(control_phrase) => control_phrase.to_string(),
        }
    }
}

/// A set of control phrases to pass information from the transcriber to a UI
// These would benefit from some eventual localization
#[derive(EnumString, IntoStaticStr, Display)]
pub enum WhisperControlPhrase {
    /// Preparing whisper for transcription
    #[strum(serialize = "[GETTING_READY]")]
    GettingReady,
    /// Whisper is set up and the transcriber loop is running to decode audio
    #[strum(serialize = "[START SPEAKING]")]
    StartSpeaking,
    /// The transcription time has exceeded its user-specified timeout boundary
    #[strum(serialize = "[TRANSCRIPTION TIMEOUT]")]
    TranscriptionTimeout,
    /// The transcription has fully ended and the final string will be returned
    #[strum(serialize = "[END TRANSCRIPTION]")]
    EndTranscription,
    /// For passing debugging messages across the channel
    #[strum(serialize = "Debug: {0}")]
    Debug(String),
}
