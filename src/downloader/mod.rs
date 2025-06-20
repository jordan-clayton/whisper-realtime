use std::fs::{self, File};
use std::path::Path;

use crate::utils::errors::RibbleWhisperError;

pub mod downloaders;

/// For downloading an object synchronously (blocking)
pub trait SyncDownload: Writable {
    fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), RibbleWhisperError>;
}

/// For downloading an object asynchronously (non-blocking, requires async runtime)
#[cfg(feature = "downloader-async")]
pub trait AsyncDownload: Writable {
    fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> impl Future<Output = Result<(), RibbleWhisperError>>;
}

/// To handle basic IO operations when downlading files
pub trait Writable {
    // If a file path does not already exist, it will be created.
    fn prepare_file_path(file_directory: &Path) -> Result<(), RibbleWhisperError> {
        if !file_directory.exists() {
            fs::create_dir_all(file_directory)?;
        }

        Ok(())
    }

    fn open_write_file(file_path: &Path) -> Result<File, RibbleWhisperError> {
        let dest = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(file_path)?;

        Ok(dest)
    }
}
