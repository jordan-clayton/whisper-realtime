// Model unit tests
// TODO: implement meaningful testing or remove this module, the old tests do not provide useful information.
#[cfg(test)]
mod model_tests {
    use whisper_realtime::whisper::model::DefaultModelType;

    #[test]
    fn test_model_file_path() {
        let home_dir_string = std::env::var_os("HOME").unwrap();
        let home_path = std::path::Path::new(&home_dir_string);

        let mut model_directory = home_path.to_path_buf();

        // NOTE: This has not been tested.
        #[cfg(target_os = "windows")]
        {
            model_directory.push("RoamingAppData");
            model_directory.push("whisper_realtime");
            model_directory.push("data");
            model_directory.push("models");
        }

        // NOTE: This has not been tested.
        #[cfg(target_os = "macos")]
        {
            model_directory.push("Library");
            model_directory.push("Application Support");
            model_directory.push("com.jordan.whisper_realtime");
            model_directory.push("models");
        }
        #[cfg(target_os = "linux")]
        {
            model_directory.push(".local");
            model_directory.push("share");
            model_directory.push("whisper_realtime");
            model_directory.push("data");
            model_directory.push("models");
        }

        let model = DefaultModelType::default()
            .to_model()
            .with_path_prefix(model_directory.as_path());

        let test_dir = model.path_prefix().to_path_buf();
        let test_dir_str = test_dir.as_os_str();
        let expected_dir_str = model_directory.as_os_str();

        assert_eq!(
            test_dir_str, expected_dir_str,
            "Path mismatch. Test: {:?}, Expected: {:?}",
            test_dir, expected_dir_str
        );
    }
    #[test]
    fn test_file_path() {
        let home_dir_string = std::env::var_os("HOME").unwrap();
        let home_path = std::path::Path::new(&home_dir_string);
        let mut model_directory = home_path.to_path_buf();

        // NOTE: This has not been tested.
        #[cfg(target_os = "windows")]
        {
            model_directory.push("RoamingAppData");
            model_directory.push("whisper_realtime");
            model_directory.push("data");
            model_directory.push("models");
        }

        // NOTE: This has not been tested.
        #[cfg(target_os = "macos")]
        {
            model_directory.push("Library");
            model_directory.push("Application Support");
            model_directory.push("com.jordan.whisper_realtime");
            model_directory.push("data");
            model_directory.push("models");
        }
        #[cfg(target_os = "linux")]
        {
            model_directory.push(".local");
            model_directory.push("share");
            model_directory.push("whisper_realtime");
            model_directory.push("data");
            model_directory.push("models");
        }

        let model = DefaultModelType::default()
            .to_model()
            .with_path_prefix(model_directory.as_path());

        model_directory.push("ggml-tiny.en.bin");
        let test_file_path = model.file_path();
        let test_file_str = test_file_path.as_os_str();
        let expected_file_str = model_directory.as_os_str();

        assert_eq!(
            test_file_str, expected_file_str,
            "File path mismatch. Test: {:?}, Expected: {:?}",
            test_file_str, expected_file_str
        );
    }

    #[test]
    fn test_url() {
        let model_type = DefaultModelType::default();

        let test_url = model_type.url();
        let test_url_str = test_url.as_str();
        const EXPECTED_URL_STR: &'static str =
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";

        assert_eq!(
            test_url_str, EXPECTED_URL_STR,
            "Url malformed. Test: {}, Expected: {}",
            test_url_str, EXPECTED_URL_STR
        )
    }
}
