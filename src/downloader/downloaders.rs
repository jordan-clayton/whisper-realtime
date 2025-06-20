#[cfg(feature = "downloader-async")]
use std::io::Write;
use std::io::{copy, Read};
use std::path::Path;

#[cfg(feature = "downloader-async")]
use bytes::Bytes;
#[cfg(feature = "downloader-async")]
use futures::StreamExt;
#[cfg(feature = "downloader-async")]
use futures_core::stream::Stream;
use url::Url;

#[cfg(feature = "downloader-async")]
use crate::downloader::AsyncDownload;
use crate::downloader::{SyncDownload, Writable};
use crate::utils::callback::{Callback, Nop};
use crate::utils::errors::RibbleWhisperError;

/// Streams in bytes (asynchronously) to download data.
/// Current progress can be obtained by supplying a Callback.
#[cfg(feature = "downloader-async")]
pub struct StreamDownloader<
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
    CB: Callback<Argument = usize>,
> {
    // Bytestream
    file_stream: S,
    // Total progress thus far
    progress: usize,
    // Total download size
    total_size: usize,
    // Optional progress callback. Default is a Nop
    progress_callback: CB,
}
#[cfg(feature = "downloader-async")]
impl<S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin> StreamDownloader<S, Nop<usize>> {
    /// Returns a StreamDownloader with the default (NOP) callback
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
    /// Returns a StreamDownloader with a set ProgressCallback
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
    /// Sets the file stream to be downloaded
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
    /// Sets an optional progress callback.
    /// To un-set a callback, supply [Nop]
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

    /// Gets the download's total size
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
    /// Downloads a given file asynchronously to the desired location. Returns Err on I/O failure.
    /// This function must be awaited and should not be called on a UI thread.
    async fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), RibbleWhisperError> {
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

/// Downloads a file synchronously (blocking).
/// Current progress can be obtained by supplying a Callback.
pub struct SyncDownloader<R: Read, CB: Callback<Argument = usize>> {
    file_stream: R,
    progress: usize,
    total_size: usize,
    progress_callback: CB,
}

impl<R: Read> SyncDownloader<R, Nop<usize>> {
    /// Returns a SyncDownloader with the default (NOP) callback
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
    /// Returns a SyncDownloader with the provided optional progress callback.
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

    /// Sets the file stream.
    pub fn with_file_stream<R2: Read>(self, file_stream: R2) -> SyncDownloader<R2, CB> {
        SyncDownloader::new_with_parameters_and_callback(
            file_stream,
            self.total_size,
            self.progress_callback,
        )
    }
    /// Sets the (optional) progress callback.
    /// To un-set the callback, supply a [Nop]
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

    /// Gets the download's total size
    pub fn total_size(&self) -> usize {
        self.total_size
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
    /// Downloads a file synchronously to the desired location. Returns Err on I/O failure.
    /// This will block the calling thread.
    fn download(
        &mut self,
        file_directory: &Path,
        file_name: &str,
    ) -> Result<(), RibbleWhisperError> {
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

/// Returns a StreamDownloader that encapsulates the request bytestream, progress,
/// and total response size.
/// Call [StreamDownloader::with_progress_callback] to set an optional callback to receive
/// updates on the number of bytes downloaded.
///
/// NOTE: This function must be awaited and should not be called on a UI thread.
/// # Returns:
/// Ok(SyncDownloader) on success, Err on a failure to either send the request or get the content length
#[cfg(feature = "downloader-async")]
pub async fn async_download_request(
    url: &str,
) -> Result<
    StreamDownloader<impl Stream<Item = Result<Bytes, reqwest::Error>>, Nop<usize>>,
    RibbleWhisperError,
> {
    let m_url = Url::parse(url)?;
    let client = reqwest::Client::new();

    let res = client.get(m_url).send().await?;

    if !res.status().is_success() {
        return Err(RibbleWhisperError::DownloadError(format!(
            "Failed to download, status code: {}",
            res.status()
        )));
    }

    let total_size = res
        .content_length()
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to get content length".to_owned(),
        ))? as usize;

    let stream = res.bytes_stream();
    // Return the appropriate streamdownloader.
    // Uh.
    Ok(StreamDownloader::new_with_parameters(stream, total_size))
}

/// Creates a SyncDownloader that encapsulates the request bytestream, progress,
/// and total response size.
/// Call [SyncDownloader::with_progress_callback] to set an optional callback to receive
/// updates on the number of bytes downloaded.
///
/// NOTE: SyncDownloaders are blocking and thus will block the calling thread.
/// # Returns:
/// Ok(SyncDownloader) on success, Err on a failure to either send the request or get the content length

pub fn sync_download_request(
    url: &str,
) -> Result<SyncDownloader<impl Read, Nop<usize>>, RibbleWhisperError> {
    let m_url = Url::parse(url)?;
    let client = reqwest::blocking::Client::new();

    let res = client.get(m_url).send()?;

    if !res.status().is_success() {
        return Err(RibbleWhisperError::DownloadError(format!(
            "Failed to download, status code: {}",
            res.status()
        )));
    }

    let total_size = res
        .content_length()
        .ok_or(RibbleWhisperError::ParameterError(
            "Failed to get content length".to_owned(),
        ))? as usize;

    Ok(SyncDownloader::new_with_parameters(res, total_size))
}
