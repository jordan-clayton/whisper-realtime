use std::io::{copy, Read, Write};
use std::path::Path;

use bytes::Bytes;
pub use futures::StreamExt;
pub use futures_core::stream::Stream;

use crate::downloader::traits::{AsyncDownload, SyncDownload, Writable};
use crate::utils::callback::{Callback, Nop};
use crate::utils::errors::WhisperRealtimeError;

/// Streamdownloader is for streaming in downloaded data from a successful request.
///
/// Use the progress callback to receive the in-progress total bytes downloaded + written
/// to the desired file path.
pub struct StreamDownloader<
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    CB: Callback<Argument = usize>,
> {
    file_stream: S,
    progress: usize,
    total_size: usize,
    progress_callback: CB,
}
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

impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin, CB: Callback<Argument = usize>>
    Writable for StreamDownloader<S, CB>
{
}

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
///
/// NOTE: download() uses std::fs::copy and is considerably faster, but will not call
/// a progress_callback.
pub struct SyncDownloader<R: Read> {
    file_stream: R,
    progress: usize,
    total_size: usize,
}

impl<R: Read> SyncDownloader<R> {
    pub fn new_with_parameters(file_stream: R, total_size: usize) -> Self {
        Self {
            file_stream,
            progress: 0,
            total_size,
        }
    }
    pub fn with_file_stream<R2: Read>(self, file_stream: R2) -> SyncDownloader<R2> {
        SyncDownloader::new_with_parameters(file_stream, self.total_size)
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

impl<R: Read> Read for SyncDownloader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let byte_read = self.file_stream.read(buf);
        if let Ok(num_bytes) = byte_read {
            self.progress += num_bytes;
        };
        byte_read
    }
}

impl<R: Read> Writable for SyncDownloader<R> {}

impl<R: Read> SyncDownload for SyncDownloader<R> {
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
        let source = &mut self.file_stream;
        copy(source, &mut dest)?;

        Ok(())
    }
}
