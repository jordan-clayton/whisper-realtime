#[cfg(test)]
#[cfg(feature = "downloader-async")]
mod downloader_tests {
    use indicatif::{ProgressBar, ProgressStyle};
    use reqwest;
    use tokio::runtime::Runtime;

    use whisper_realtime::downloader;
    use whisper_realtime::downloader::AsyncDownload;
    use whisper_realtime::downloader::SyncDownload;
    use whisper_realtime::utils::callback::ProgressCallback;
    use whisper_realtime::utils::errors::WhisperRealtimeError;
    use whisper_realtime::whisper::model::DefaultModelType;

    fn delete_model(file_path: &std::path::Path) -> std::io::Result<()> {
        std::fs::remove_file(file_path)
    }

    #[test]
    fn test_async_download() {
        let path = std::env::current_dir().unwrap().join("data").join("models");
        let model_type = DefaultModelType::default();
        let model = model_type.to_model().with_path_prefix(path.as_path());
        let file_path = model.file_path();
        let file_name = model.file_name();
        let file_directory = model.path_prefix();

        if model.exists_in_directory() {
            let deleted = delete_model(file_path.as_path());
            assert!(
                deleted.is_ok(),
                "Failed to delete model: {}",
                deleted.unwrap_err()
            )
        }

        assert!(
            !model.exists_in_directory(),
            "File still exists in directory."
        );

        let url = model_type.url();

        // Tokio runtime for block_on.
        let rt = Runtime::new().unwrap();
        let handle = rt.handle();

        // NOTE: callback url is public and can be set after creating the struct by using the builder.
        let stream_downloader = downloader::downloaders::async_download_request(url.as_str());

        // Run the future -> At this time, tokio is not being used for async and this cannot be awaited in testing.
        let stream_downloader = handle.block_on(stream_downloader);

        // Ensure proper struct creation
        assert!(
            stream_downloader.is_ok(),
            "{}",
            stream_downloader.err().unwrap()
        );

        let stream_downloader = stream_downloader.unwrap();

        // Initiate a progress bar.
        let pb = ProgressBar::new(stream_downloader.total_size() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
            .progress_chars("#>-")
        );

        let pb_c = pb.clone();
        let progress_callback_function = move |n: usize| {
            pb_c.set_position(n as u64);
        };
        let progress_callback = ProgressCallback::new(progress_callback_function);

        let mut stream_downloader = stream_downloader.with_progress_callback(progress_callback);

        let download = stream_downloader.download(file_directory, file_name);

        let download = handle.block_on(download);

        assert!(
            download.is_ok(),
            "{}",
            format!("{}", download.err().unwrap())
        );

        assert!(
            model.exists_in_directory(),
            "Model not successfully downloaded or file path incorrect."
        );
    }

    #[test]
    fn test_sync_download() {
        let path = std::env::current_dir().unwrap().join("data").join("models");
        let model_type = DefaultModelType::default();
        let model = model_type.to_model().with_path_prefix(path.as_path());
        let file_path = model.file_path();
        let file_name = model.file_name();
        let file_directory = model.path_prefix();

        if model.exists_in_directory() {
            let deleted = delete_model(file_path.as_path());
            assert!(
                deleted.is_ok(),
                "Failed to delete model: {}",
                deleted.unwrap_err()
            )
        }

        assert!(
            !model.exists_in_directory(),
            "File still exists in directory."
        );

        let url = model_type.url();
        let client = reqwest::blocking::Client::new();

        // NOTE: callback url is public and can be set after creating the struct by using the builder.
        let sync_downloader = downloader::downloaders::sync_download_request(url.as_str());

        // Ensure proper struct creation + download.
        assert!(
            sync_downloader.is_ok(),
            "{}",
            sync_downloader.err().unwrap()
        );

        let sync_downloader = sync_downloader.unwrap();

        // Initiate a progress bar.
        let pb = ProgressBar::new(sync_downloader.total_size() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
            .progress_chars("#>-")
        );

        let pb_c = pb.clone();
        let progress_callback_function = move |n: usize| {
            pb_c.set_position(n as u64);
        };
        let progress_callback = ProgressCallback::new(progress_callback_function);

        let mut sync_downloader = sync_downloader.with_progress_callback(progress_callback);

        // File copy:
        let download: Result<(), WhisperRealtimeError> =
            sync_downloader.download(file_directory, file_name);

        assert!(download.is_ok(), "{}", format!("{}", download.unwrap_err()));

        assert!(
            model.exists_in_directory(),
            "Model not successfully copied or file path incorrect."
        );
    }
}
