#[derive(Debug, Clone)]
pub struct TranscriptionError {
    reason: String,
}

impl TranscriptionError {
    pub fn new() -> TranscriptionError {
        TranscriptionError {
            reason: String::from("Unknown Error"),
        }
    }

    pub fn new_with_reason(reason: String) -> TranscriptionError {
        TranscriptionError { reason }
    }
    pub fn cause(&self) -> String {
        self.reason.clone()
    }
}

impl std::fmt::Display for TranscriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}
