// Model unit tests
// TODO: these will need to be refactored once the implementation has been redesigned
// TODO: also, make the tests a little more useful
#[cfg(test)]
mod model_tests {
    use whisper_realtime::whisper::model::OldModel;

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
        }

        // NOTE: This has not been tested.
        #[cfg(target_os = "macos")]
        {
            model_directory.push("Library");
            model_directory.push("Application Support");
            model_directory.push("com.jordan.whisper_realtime");
        }
        #[cfg(target_os = "linux")]
        {
            model_directory.push(".local");
            model_directory.push("share");
            model_directory.push("whisper_realtime");
            model_directory.push("data");
        }

        let model: OldModel = OldModel::default().with_data_dir(model_directory.clone());

        model_directory.push("models");

        let test_dir = model.model_directory();
        let test_dir = test_dir.as_os_str();
        let expected_dir = model_directory.as_os_str();

        assert_eq!(test_dir, expected_dir,);
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
        }

        // NOTE: This has not been tested.
        #[cfg(target_os = "macos")]
        {
            model_directory.push("Library");
            model_directory.push("Application Support");
            model_directory.push("com.jordan.whisper_realtime");
            model_directory.push("data");
        }
        #[cfg(target_os = "linux")]
        {
            model_directory.push(".local");
            model_directory.push("share");
            model_directory.push("whisper_realtime");
            model_directory.push("data");
        }

        let model: OldModel = OldModel::default().with_data_dir(model_directory.clone());

        model_directory.push("models");
        model_directory.push("tiny.en.bin");
        let test_file_path = model.file_path();
        let test_file = test_file_path.as_os_str();
        let expected_file = model_directory.as_os_str();

        assert_eq!(test_file, expected_file,);
    }

    #[test]
    fn test_url() {
        let model: OldModel = OldModel::default();

        let test_url = model.url();
        let test_url = test_url.as_str();

        assert_eq!(
            test_url,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
        )
    }
}
