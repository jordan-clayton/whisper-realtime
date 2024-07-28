use std::io::Read;

use bytes::Bytes;
use futures_core::stream::Stream;
use reqwest::Url;

use crate::downloader::downloader::{StreamDownloader, SyncDownloader};
use crate::errors::{WhisperRealtimeError, WhisperRealtimeErrorType};

/// Returns a StreamDownloader struct encapsulating the request bytestream, progress, total
/// response size, and an optional callback function to receive progress updates.
/// This function must be awaited and should not be called on a UI thread.
async fn async_download_request<CB: Fn(usize)>(
    client: &reqwest::Client,
    url: &str,
    progress_callback: Option<CB>,
) -> Result<
    StreamDownloader<impl Stream<Item = Result<Bytes, reqwest::Error>>, CB>,
    WhisperRealtimeError,
> {
    let m_url = Url::parse(url);
    if let Err(e) = m_url {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::DownloadError,
            format!("Failed to parse url: {}, Error: {:?} ", url, e),
        ));
    };

    let m_url = m_url.unwrap();

    let res = client
        .get(m_url)
        .send()
        .await
        .or(Err(format!("Failed to GET from {}", url)));

    if let Err(e) = res {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::DownloadError,
            format!("{} ", e),
        ));
    };

    let res = res.unwrap();

    if !res.status().is_success() {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::DownloadError,
            format!(
                "Failed to download URL: {}, Error Code: {}",
                url,
                res.status()
            ),
        ));
    }

    let total_size = res
        .content_length()
        .ok_or(format!("Failed to get content length from {}", url));

    if let Err(e) = total_size {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::DownloadError,
            format!("{} ", e),
        ));
    };

    let total_size = total_size.unwrap() as usize;

    let stream = res.bytes_stream();
    Ok(StreamDownloader::new(stream, total_size, progress_callback))
}

/// Returns a SyncDownloader struct encapsulating the downloaded response, progress,
/// total response size, and an optional callback function provided to receive progress updates.
/// This is strictly for synchronous downloading and will block the calling thread.
/// It is recommended to call this function on a separate thread if other work needs to be performed.

fn sync_download_request<CB: Fn(usize)>(
    client: &reqwest::blocking::Client,
    url: &str,
    progress_callback: Option<CB>,
) -> Result<SyncDownloader<impl Read, CB>, WhisperRealtimeError> {
    let m_url = Url::parse(url);
    if let Err(e) = m_url {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::DownloadError,
            format!("Failed to parse url: {}, Error: {:?} ", url, e),
        ));
    };

    let m_url = m_url.unwrap();
    let res = client.get(m_url).send();

    if let Err(e) = res {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::DownloadError,
            format!("Failed to GET from {}, Error: {:?} ", url, e),
        ));
    };

    let res = res.unwrap();

    if !res.status().is_success() {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::DownloadError,
            format!(
                "Failed to download URL: {}, Error Code: {}",
                url,
                res.status()
            ),
        ));
    };

    let total_size = res
        .content_length()
        .ok_or(format!("Failed to get content length from {}", url));

    if let Err(e) = total_size {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::DownloadError,
            format!("{} ", e),
        ));
    };

    let total_size = total_size.unwrap() as usize;
    Ok(SyncDownloader::new(res, total_size, progress_callback))
}
