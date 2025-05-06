use std::io::{copy, Read};
#[cfg(feature = "downloader-async")]
use std::io::Write;
use std::path::Path;

#[cfg(feature = "downloader-async")]
use bytes::Bytes;
#[cfg(feature = "downloader-async")]
pub use futures::StreamExt;
#[cfg(feature = "downloader-async")]
pub use futures_core::stream::Stream;

use crate::downloader::traits::{SyncDownload, Writable};
#[cfg(feature = "downloader-async")]
use crate::downloader::traits::AsyncDownload;
use crate::utils::callback::{Callback, Nop};
use crate::utils::errors::WhisperRealtimeError;

/// Streamdownloader is for streaming in downloaded data from a successful request.
///
/// Use the progress callback to receive the in-progress total bytes downloaded + written
/// to the desired file path.
#[cfg(feature = "downloader-async")]
pub struct StreamDownloader<
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    CB: Callback<Argument = usize>,
> {
    file_stream: S,
    progress: usize,
    total_size: usize,
    progress_callback: CB,
}
#[cfg(feature = "downloader-async")]
impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin> StreamDownloader<S, Nop<usize>> {
    pub fn new_with_parameters(file_stream: S, total_size: usize) -> Self {
        Self {
            file_stream,
            progress: 0,
            total_size,
            progress_callback: Nop::new(),
        }
    }
}

#[cfg(feature = "downloader-async")]
impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin, CB: Callback<Argument = usize>>
    StreamDownloader<S, CB>
{
    pub fn new_with_parameters_and_callback(
        file_stream: S,
        total_size: usize,
        progress_callback: CB,
    ) -> Self {
        Self {
            file_stream,
            progress: 0,
            total_size,
            progress_callback,
        }
    }
    pub fn with_file_stream<S2: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin>(
        self,
        file_stream: S2,
    ) -> StreamDownloader<S2, CB> {
        StreamDownloader::new_with_parameters_and_callback(
            file_stream,
            self.total_size,
            self.progress_callback,
        )
    }
    pub fn with_progress_callback<C: Callback<Argument = usize>>(
        self,
        progress_callback: C,
    ) -> StreamDownloader<S, C> {
        StreamDownloader::new_with_parameters_and_callback(
            self.file_stream,
            self.total_size,
            progress_callback,
        )
    }
    pub fn with_total_size(mut self, total_size: usize) -> Self {
        self.total_size = total_size;
        return self;
    }
    pub fn total_size(&self) -> usize {
        self.total_size
    }
}

#[cfg(feature = "downloader-async")]
impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin, CB: Callback<Argument = usize>>
    Writable for StreamDownloader<S, CB>
{
}

#[cfg(feature = "downloader-async")]
impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin, CB: Callback<Argument = usize>>
    AsyncDownload for StreamDownloader<S, CB>
{
    async fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), WhisperRealtimeError> {
        Self::prepare_file_path(file_directory)?;

        let mut destination = file_directory.to_path_buf();
        destination.push(file_name);

        let mut dest = Self::open_write_file(&destination)?;

        let stream = &mut self.file_stream;

        while let Some(next) = stream.next().await {
            let buf = next?;
            dest.write_all(&buf)?;

            let mut cur_progress = self.progress;

            cur_progress = std::cmp::min(cur_progress + (buf.len()), self.total_size);
            self.progress = cur_progress;

            // Update the UI with the current progress
            self.progress_callback.call(self.progress);
        }
        Ok(())
    }
}

/// SyncDownloader is for making a synchronous copy of a downloaded request to the filesystem.
/// It does not provide download progress, but will provide write progress.
///
/// Use the progress callback to receive the total number of bytes copied per read while copying
/// the download to the filesystem using download_with_progress()
pub struct SyncDownloader<R: Read, CB: Callback<Argument = usize>> {
    file_stream: R,
    progress: usize,
    total_size: usize,
    progress_callback: CB,
}

impl<R: Read> SyncDownloader<R, Nop<usize>> {
    pub fn new_with_parameters(file_stream: R, total_size: usize) -> Self {
        Self {
            file_stream,
            progress: 0,
            total_size,
            progress_callback: Nop::new(),
        }
    }
}

impl<R: Read, CB: Callback<Argument = usize>> SyncDownloader<R, CB> {
    pub fn new_with_parameters_and_callback(
        file_stream: R,
        total_size: usize,
        progress_callback: CB,
    ) -> Self {
        Self {
            file_stream,
            progress: 0,
            total_size,
            progress_callback,
        }
    }
    pub fn with_file_stream<R2: Read>(self, file_stream: R2) -> SyncDownloader<R2, CB> {
        SyncDownloader::new_with_parameters_and_callback(
            file_stream,
            self.total_size,
            self.progress_callback,
        )
    }

    pub fn with_progress_callback<C: Callback<Argument = usize>>(
        self,
        progress_callback: C,
    ) -> SyncDownloader<R, C> {
        SyncDownloader::new_with_parameters_and_callback(
            self.file_stream,
            self.total_size,
            progress_callback,
        )
    }
    pub fn with_total_size(mut self, total_size: usize) -> Self {
        self.total_size = total_size;
        self
    }
    pub fn total_size(&self) -> usize {
        self.total_size
    }
    pub fn progress(&self) -> usize {
        self.progress
    }
}

impl<R: Read, CB: Callback<Argument = usize>> Read for SyncDownloader<R, CB> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let byte_read = self.file_stream.read(buf);
        if let Ok(num_bytes) = byte_read {
            self.progress += num_bytes;

            // Update the UI with the current progress
            self.progress_callback.call(self.progress);
        };
        byte_read
    }
}

impl<R: Read, CB: Callback<Argument = usize>> Writable for SyncDownloader<R, CB> {}

impl<R: Read, CB: Callback<Argument = usize>> SyncDownload for SyncDownloader<R, CB> {
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

        let mut dest = Self::open_write_file(&destination)?;
        copy(self, &mut dest)?;

        Ok(())
    }
}
