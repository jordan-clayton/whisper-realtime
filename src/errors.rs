use std::{error, fmt};

#[derive(Debug, Clone)]
pub struct WhisperRealtimeError {
    error_type: WhisperRealtimeErrorType,
    pub reason: String,
}

#[derive(Debug, Clone, Copy)]
pub enum WhisperRealtimeErrorType {
    TranscriptionError,
    DownloadError,
    WriteError,
    ParameterError,
    Unknown,
}

impl fmt::Display for WhisperRealtimeErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WhisperRealtimeErrorType::TranscriptionError => {
                write!(f, "TranscriptionError")
            }
            WhisperRealtimeErrorType::DownloadError => {
                write!(f, "DownloadError")
            }
            WhisperRealtimeErrorType::WriteError => {
                write!(f, "WriteError")
            }
            WhisperRealtimeErrorType::ParameterError => {
                write!(f, "ParameterErrorError")
            }
            WhisperRealtimeErrorType::Unknown => {
                write!(f, "UnknownError")
            }
        }
    }
}

impl WhisperRealtimeError {
    pub fn new(error_type: WhisperRealtimeErrorType, reason: String) -> Self {
        Self { error_type, reason }
    }

    pub fn cause(&self) -> String {
        format!("{}: {}", self.error_type, self.reason)
    }
}

impl fmt::Display for WhisperRealtimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.cause())
    }
}

impl error::Error for WhisperRealtimeError {}
