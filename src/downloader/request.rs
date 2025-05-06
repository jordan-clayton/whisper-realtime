use std::io::Read;

#[cfg(feature = "downloader-async")]
pub use bytes::Bytes;
#[cfg(feature = "downloader-async")]
pub use futures_core::stream::Stream;
pub use reqwest::{self, Url};

#[cfg(feature = "downloader-async")]
use crate::downloader::downloaders::StreamDownloader;
use crate::downloader::downloaders::SyncDownloader;
use crate::utils::callback::Nop;
use crate::utils::errors::WhisperRealtimeError;

/// Returns a StreamDownloader struct encapsulating the request bytestream, progress,
/// and total response size
/// Use .with_progress_callback() to set an optional progress callback that operates on the download's
/// current progress (ie. total bytes downloaded thus far)
/// This function must be awaited and should not be called on a UI thread.
#[cfg(feature = "downloader-async")]
pub async fn async_download_request(
    client: &reqwest::Client,
    url: &str,
) -> Result<
    StreamDownloader<impl Stream<Item = Result<Bytes, reqwest::Error>>, Nop<usize>>,
    WhisperRealtimeError,
> {
    let m_url = Url::parse(url)?;

    let res = client.get(m_url).send().await?;

    if !res.status().is_success() {
        return Err(WhisperRealtimeError::DownloadError(format!(
            "Failed to download, status code: {}",
            res.status()
        )));
    }

    let total_size = res
        .content_length()
        .ok_or(WhisperRealtimeError::ParameterError(
            "Failed to get content length".to_owned(),
        ))? as usize;

    let stream = res.bytes_stream();
    // Return the appropriate streamdownloader.
    // Uh.
    Ok(StreamDownloader::new_with_parameters(stream, total_size))
}

/// Returns a SyncDownloader struct encapsulating the downloaded response, progress,
/// and total response size.
/// Use .with_progress_callback() to set an optional progress callback that operates on the download's
/// current progress (ie. total bytes downloaded thus far)
/// This is strictly for synchronous downloading and will block the calling thread.
/// It is recommended to call this function on a separate thread if other work needs to be performed.

pub fn sync_download_request(
    client: &reqwest::blocking::Client,
    url: &str,
) -> Result<SyncDownloader<impl Read, Nop<usize>>, WhisperRealtimeError> {
    let m_url = Url::parse(url)?;

    let res = client.get(m_url).send()?;

    if !res.status().is_success() {
        return Err(WhisperRealtimeError::DownloadError(format!(
            "Failed to download, status code: {}",
            res.status()
        )));
    }

    let total_size = res
        .content_length()
        .ok_or(WhisperRealtimeError::ParameterError(
            "Failed to get content length".to_owned(),
        ))? as usize;

    Ok(SyncDownloader::new_with_parameters(res, total_size))
}
