#[cfg(test)]
mod downloader_tests {
    use indicatif::{ProgressBar, ProgressStyle};
    use reqwest;
    use tokio::runtime::Runtime;

    use whisper_realtime::{
        downloader,
        downloader::traits::{AsyncDownload, SyncDownload},
        errors::WhisperRealtimeError,
        model,
    };

    #[test]
    #[ignore]
    fn test_async_download() {
        let model: model::Model = model::Model::default();

        // Delete the model
        model.delete();

        assert!(!model.is_downloaded());

        let url = model.url();
        let url = url.as_str();
        let client = reqwest::Client::new();

        // Tokio runtime for block_on.
        let rt = Runtime::new().unwrap();
        let handle = rt.handle();

        // NOTE: callback url is public and can be set after creating the struct.
        let stream_downloader = downloader::request::async_download_request(&client, url, None);

        // Run the future -> At this time, tokio is not being used for async and this cannot be awaited in testing.
        let stream_downloader = handle.block_on(stream_downloader);

        // Ensure proper struct creation
        assert!(
            stream_downloader.is_ok(),
            "{}",
            format!("{}", stream_downloader.err().unwrap().cause())
        );

        let mut stream_downloader = stream_downloader.unwrap();

        // Initiate a progress bar.
        let pb = ProgressBar::new(stream_downloader.total_size as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
            .progress_chars("#>-")
        );

        let pb_c = pb.clone();

        stream_downloader.progress_callback = Some(move |n: usize| {
            pb_c.set_position(n as u64);
        });

        // File Path:
        let file_directory = model.model_directory();
        let file_directory = file_directory.as_path();
        let file_name = model.model_file_name();

        let download = stream_downloader.download(file_directory, file_name);

        let download = handle.block_on(download);

        assert!(
            download.is_ok(),
            "{}",
            format!("{}", download.err().unwrap().cause())
        );

        assert!(
            model.is_downloaded(),
            "Model not successfully downloaded or file path incorrect."
        );
    }

    #[test]
    #[ignore]
    fn test_sync_download() {
        let mut model: model::Model = model::Model::default();
        // model.model_type = model::ModelType::MediumEn;

        // Delete the model
        model.delete();

        assert!(!model.is_downloaded());

        let url = model.url();
        let url = url.as_str();
        let client = reqwest::blocking::Client::new();

        // NOTE: callback url is public and can be set after creating the struct.
        let sync_downloader = downloader::request::sync_download_request(&client, url, None);

        // Ensure proper struct creation + download.
        assert!(
            sync_downloader.is_ok(),
            "{}",
            format!("{}", sync_downloader.err().unwrap().cause())
        );

        let mut sync_downloader = sync_downloader.unwrap();

        // File Path:
        let file_directory = model.model_directory();
        let file_directory = file_directory.as_path();
        let file_name = model.model_file_name();

        // File copy:
        let download: Result<(), WhisperRealtimeError> =
            sync_downloader.download(file_directory, file_name);

        assert!(
            download.is_ok(),
            "{}",
            format!("{}", download.err().unwrap().cause())
        );

        assert!(
            model.is_downloaded(),
            "Model not successfully copied or file path incorrect."
        );
    }
}
