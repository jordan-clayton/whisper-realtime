// TODO: Copy information from README.md (ie. examples, feature flags) over here to generate proper
// docs (USE MARKDOWN SYNTAX)
pub mod audio;
#[cfg(feature = "downloader")]
pub mod downloader;
pub mod transcriber;
pub mod utils;
pub mod whisper;
