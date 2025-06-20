#![doc = include_str!("../README.md")]
// TODO: either re-export major structs/traits/etc. here, or write a prelude
pub mod audio;
#[cfg(feature = "downloader")]
pub mod downloader;
pub mod transcriber;
pub mod utils;
pub mod whisper;
