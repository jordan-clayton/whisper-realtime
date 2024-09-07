use std::{
    fs::{self, File},
    path::Path,
};

use crate::errors::{WhisperRealtimeError, WhisperRealtimeErrorType};

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
            let created_directory = fs::create_dir_all(file_directory);
            if let Err(_) = created_directory {
                return Err(WhisperRealtimeError::new(
                    WhisperRealtimeErrorType::WriteError,
                    String::from("Failed to create download directory"),
                ));
            }
        }

        Ok(())
    }

    fn open_write_file(file_path: &Path) -> Result<File, WhisperRealtimeError> {
        let dest = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(file_path);

        if let Err(e) = dest {
            return Err(WhisperRealtimeError::new(
                WhisperRealtimeErrorType::WriteError,
                format!("Failed to open destination file, Error: {:?}", e),
            ));
        };

        Ok(dest.unwrap())
    }
}
