mod common;
#[cfg(test)]
#[cfg(feature = "downloader-async")]
mod downloader_tests {
    use indicatif::{ProgressBar, ProgressStyle};
    use tokio::runtime::Runtime;

    use crate::common::prep_model_bank;
    use ribble_whisper::downloader;
    use ribble_whisper::downloader::AsyncDownload;
    use ribble_whisper::downloader::SyncDownload;
    use ribble_whisper::utils::callback::RibbleWhisperCallback;
    use ribble_whisper::whisper::model::{DefaultModelType, ModelBank, ModelRetriever};

    fn delete_model(file_path: &std::path::Path) -> std::io::Result<()> {
        std::fs::remove_file(file_path)
    }

    #[test]
    fn test_async_download() {
        // NOTE: When running tests in parallel, these need to be different
        let model_type = DefaultModelType::Tiny;
        let (model_bank, model_id) = prep_model_bank(model_type);
        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "IO error in model bank in async download test."
        );

        if exists.unwrap() {
            let file_path = model_bank.retrieve_model_path(model_id);
            assert!(file_path.is_some(), "Model bank not returning file path.");
            let file_path = file_path.unwrap();
            let deleted = delete_model(file_path.as_path());
            assert!(
                deleted.is_ok(),
                "Failed to delete model. Error: {}, Path: {:?}",
                deleted.unwrap_err(),
                file_path,
            )
        }

        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "IO error in model bank in async download test after deletion."
        );

        assert!(!exists.unwrap(), "File still exists in directory.");

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
        let progress_callback = RibbleWhisperCallback::new(progress_callback_function);

        let mut stream_downloader = stream_downloader.with_progress_callback(progress_callback);
        let file_path = model_bank.model_directory();
        let file_name = model_bank
            .retrieve_model(model_id)
            .and_then(|model| Some(model.file_name()));
        assert!(
            file_name.is_some(),
            "Model bank failed to return a model via ID in async download test"
        );
        let file_name = file_name.unwrap();

        let download = stream_downloader.download(file_path, file_name);

        let download = handle.block_on(download);

        assert!(
            download.is_ok(),
            "{}",
            format!("{}", download.err().unwrap())
        );

        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "File IO error when checking model in folder after downloading async."
        );

        assert!(
            exists.unwrap(),
            "Model not successfully downloaded or implementation is busted."
        );
    }

    #[test]
    fn test_sync_download() {
        let model_type = DefaultModelType::TinyEn;
        let (model_bank, model_id) = prep_model_bank(model_type);

        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "IO error in model bank in sync download test."
        );

        if exists.unwrap() {
            let file_path = model_bank.retrieve_model_path(model_id);
            assert!(file_path.is_some(), "Model bank not returning file path.");
            let file_path = file_path.unwrap();
            let deleted = delete_model(file_path.as_path());
            assert!(
                deleted.is_ok(),
                "Failed to delete model. Error: {}, Path: {:?}",
                deleted.unwrap_err(),
                file_path,
            )
        }

        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "IO error in model bank in sync download test."
        );

        assert!(!exists.unwrap(), "File still exists in directory.");

        let url = model_type.url();

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
        let progress_callback = RibbleWhisperCallback::new(progress_callback_function);

        let mut sync_downloader = sync_downloader.with_progress_callback(progress_callback);

        let file_path = model_bank.model_directory();
        let file_name = model_bank
            .retrieve_model(model_id)
            .and_then(|model| Some(model.file_name()));
        assert!(
            file_name.is_some(),
            "Model bank failed to return a model via ID in async download test"
        );

        let file_name = file_name.unwrap();

        // File copy:
        let download = sync_downloader.download(file_path, file_name);

        assert!(download.is_ok(), "{}", format!("{}", download.unwrap_err()));

        let exists = model_bank.model_exists_in_storage(model_id);
        assert!(
            exists.is_ok(),
            "IO error in model bank in sync download test after downloading."
        );

        assert!(
            exists.unwrap(),
            "Model not successfully copied or file path incorrect."
        );
    }
}
