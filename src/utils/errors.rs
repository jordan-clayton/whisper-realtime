use thiserror::Error;

#[derive(Debug, Error)]
pub enum WhisperRealtimeError {
    #[error("Transcription Error {0}")]
    TranscriptionError(String),
    #[error("Reqwest Error {0}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("Download Error {0}")]
    DownloadError(String),
    #[error("Write Error {0}")]
    WriteError(String),
    #[error("Parameter Error {0}")]
    ParameterError(String),
    #[error("ResampleError: {0}")]
    ResampleError(#[from] rubato::ResampleError),
    #[error("ResamplerConstructionError: {0}")]
    ResamplerConstructionError(#[from] rubato::ResamplerConstructionError),
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
}
