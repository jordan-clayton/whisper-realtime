use crate::errors;

// Type alias for callback fn -> for updating UI when downloading.
pub trait DownloadCallback = Fn(usize);
pub trait Downloader<CB>
where
    CB: DownloadCallback,
{
    fn download(
        url: &str,
        progress_callback: Option<CB>,
    ) -> Result<(), errors::WhisperRealtimeError>;
}
