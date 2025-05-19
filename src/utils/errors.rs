use thiserror::Error;

#[derive(Debug, Error)]
pub enum WhisperRealtimeError {
    // TODO; if TranscriptionError goes unused, remove this.
    #[error("Transcription Error {0}")]
    TranscriptionError(String),
    // Called on a data-channel failure
    // Contains the message so it can be recovered.
    #[error("{0}")]
    TranscriptionSenderError(String),
    #[error("Write Error {0}")]
    WriteError(String),
    #[error("Parameter Error {0}")]
    ParameterError(String),
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("Decode error: {0}")]
    DecodeError(#[from] symphonia::core::errors::Error),
    #[error("Unknown Error {0}")]
    Unknown(String),
    #[error("Parse Error {0}")]
    ParseError(#[from] std::string::ParseError),
    #[error("UrlParse Error {0}")]
    UrlParseError(#[from] url::ParseError),
    #[error("Whisper Error {0}")]
    WhisperError(#[from] whisper_rs::WhisperError),
    #[cfg(feature = "resampler")]
    #[error("ResampleError: {0}")]
    ResampleError(#[from] rubato::ResampleError),
    #[cfg(feature = "resampler")]
    #[error("ResamplerConstructionError: {0}")]
    ResamplerConstructionError(#[from] rubato::ResamplerConstructionError),
    #[cfg(feature = "downloader")]
    #[error("Reqwest Error {0}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("Download Error {0}")]
    #[cfg(feature = "downloader")]
    DownloadError(String),
    #[error("JSON Parse Error {0}")]
    #[cfg(feature = "integrity")]
    JsonParseError(#[from] serde_json::Error),
}
