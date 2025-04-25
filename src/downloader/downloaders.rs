use std::{
    io::{copy, Read, Write},
    path::Path,
};

use bytes::Bytes;
pub use futures::StreamExt;
pub use futures_core::stream::Stream;

use crate::{
    downloader::traits::{AsyncDownload, SyncDownload, Writable},
    errors::{WhisperRealtimeError, WhisperRealtimeErrorType},
};

// TODO: make this optional, as part of extras. These are out of scope.

/// Streamdownloader is for streaming in downloaded data from a successful request.
///
/// Use the progress callback to receive the in-progress total bytes downloaded + written
/// to the desired file path.
pub struct StreamDownloader<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin, CB: Fn(usize)>
{
    file_stream: S,
    progress: usize,
    pub total_size: usize,
    pub progress_callback: Option<CB>,
}

impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin, CB: Fn(usize)>
    StreamDownloader<S, CB>
{
    pub fn new(file_stream: S, total_size: usize, progress_callback: Option<CB>) -> Self {
        Self {
            file_stream,
            progress: 0,
            total_size,
            progress_callback,
        }
    }

    // pub fn get_progress(&self) -> usize {
    //     self.progress
    // }
}

impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin, CB: Fn(usize)> Writable
    for StreamDownloader<S, CB>
{
}

impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin, CB: Fn(usize)> AsyncDownload
    for StreamDownloader<S, CB>
{
    async fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), WhisperRealtimeError> {
        let path_available = Self::prepare_file_path(file_directory);
        if let Err(e) = path_available {
            return Err(e);
        }

        let mut destination = file_directory.to_path_buf();
        destination.push(file_name);

        let dest = Self::open_write_file(&destination);
        if let Err(e) = dest {
            return Err(e);
        }

        let mut dest = dest.unwrap();
        let stream = &mut self.file_stream;

        while let Some(next) = stream.next().await {
            let chunk = next.or(Err(WhisperRealtimeError::new(
                WhisperRealtimeErrorType::DownloadError,
                String::from("Error while downloading file"),
            )));

            if let Err(e) = chunk {
                return Err(e);
            }

            let chunk = chunk.unwrap();

            let write_success = dest.write_all(&chunk);
            if let Err(_) = write_success {
                return Err(WhisperRealtimeError::new(
                    WhisperRealtimeErrorType::DownloadError,
                    String::from("Failed to write to file"),
                ));
            }

            let mut cur_progress = self.progress;

            cur_progress = std::cmp::min(cur_progress + (chunk.len()), self.total_size);
            self.progress = cur_progress;

            // Update the UI with the current progress
            if let Some(callback) = &self.progress_callback {
                callback(self.progress);
            }
        }

        Ok(())
    }
}

/// SyncDownloader is for making a synchronous copy of a downloaded request to the filesystem.
/// It does not provide download progress, but will provide write progress.
///
/// Use the progress callback to receive the total number of bytes copied per read while copying
/// the download to the filesystem using download_with_progress()
///
/// NOTE: download() uses std::fs::copy and is considerably faster, but will not call
/// the progress_callback.
pub struct SyncDownloader<R: Read, CB: Fn(usize)> {
    file_stream: R,
    progress: usize,
    pub total_size: usize,
    pub progress_callback: Option<CB>,
}

// NOTE: This needs to be implemented, but gets optimized out by the compiler in favour of a syscall.
impl<R: Read, CB: Fn(usize)> SyncDownloader<R, CB> {
    pub fn new(file_stream: R, total_size: usize, progress_callback: Option<CB>) -> Self {
        Self {
            file_stream,
            progress: 0,
            total_size,
            progress_callback,
        }
    }

    // pub fn get_progress(&self) -> usize {
    //     self.progress
    // }
}

impl<R: Read, CB: Fn(usize)> Read for SyncDownloader<R, CB> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let byte_read = self.file_stream.read(buf);
        if let Ok(num_bytes) = byte_read {
            self.progress += num_bytes;
            if let Some(callback) = &self.progress_callback {
                callback(self.progress)
            }
        };
        byte_read
    }
}

impl<R: Read, CB: Fn(usize)> Writable for SyncDownloader<R, CB> {}

impl<R: Read, CB: Fn(usize)> SyncDownload for SyncDownloader<R, CB> {
    fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), WhisperRealtimeError> {
        let path_available = Self::prepare_file_path(file_directory);
        if let Err(e) = path_available {
            return Err(e);
        }

        let mut destination = file_directory.to_path_buf();
        destination.push(file_name);

        let dest = Self::open_write_file(&destination);
        if let Err(e) = dest {
            return Err(e);
        }

        let mut dest = dest.unwrap();
        let source = &mut self.file_stream;
        let downloaded = copy(source, &mut dest);

        if let Err(_) = downloaded {
            return Err(WhisperRealtimeError::new(
                WhisperRealtimeErrorType::DownloadError,
                String::from("Failed to write file"),
            ));
        }

        Ok(())
    }

    fn download_with_progress(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), WhisperRealtimeError> {
        let path_available = Self::prepare_file_path(file_directory);
        if let Err(e) = path_available {
            return Err(e);
        }

        let mut destination = file_directory.to_path_buf();
        destination.push(file_name);

        let dest = Self::open_write_file(&destination);
        if let Err(e) = dest {
            return Err(e);
        }

        let mut dest = dest.unwrap();
        let source = &mut self.file_stream;

        // This will call the callback.
        let downloaded = std::io::copy(source, &mut dest);

        if let Err(_) = downloaded {
            return Err(WhisperRealtimeError::new(
                WhisperRealtimeErrorType::DownloadError,
                String::from("Failed to write file"),
            ));
        }

        Ok(())
    }
}
