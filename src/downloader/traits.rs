use std::fs::{self, File};
use std::path::Path;

use crate::errors::WhisperRealtimeError;

pub trait SyncDownload: Writable {
    fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), WhisperRealtimeError>;
    fn download_with_progress(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), WhisperRealtimeError>;
}

pub trait AsyncDownload: Writable {
    fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> impl std::future::Future<Output = Result<(), WhisperRealtimeError>>;
}

pub trait Writable {
    // If a file path does not already exist, it will be created.
    fn prepare_file_path(file_directory: &Path) -> Result<(), WhisperRealtimeError> {
        if !file_directory.exists() {
            fs::create_dir_all(file_directory)?;
        }

        Ok(())
    }

    fn open_write_file(file_path: &Path) -> Result<File, WhisperRealtimeError> {
        let dest = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(file_path)?;

        Ok(dest)
    }
}
