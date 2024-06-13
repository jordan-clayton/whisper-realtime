// Model unit test
#[cfg(test)]
mod model_tests {
    use whisper_realtime_gui::model::Model;

    #[test]
    fn test_model_file_path() {
        let model: Model = Model::default();
        let home_dir_string = std::env::var_os("HOME").unwrap();
        let home_path = std::path::Path::new(&home_dir_string);
        let mut model_directory = home_path.to_path_buf();
        // I don't actually know if this is correct.
        // TODO: support Windows
        #[cfg(target_os = "windows")]
        {
            model_directory.push("RoamingAppData");
            model_directory.push("WhisperGUI");
            model_directory.push("data");
            model_directory.push("models");
        }
        #[cfg(target_os = "macos")]
        {
            model_directory.push("Library");
            model_directory.push("Application Support");
            model_directory.push("com.jordan.WhisperGUI");
            model_directory.push("models");
        }
        #[cfg(target_os = "linux")]
        {
            model_directory.push(".local");
            model_directory.push("share");
            model_directory.push("whispergui");
            model_directory.push("models");
        }
        let test_dir = model.model_directory();
        let test_dir = test_dir.as_os_str();
        let expected_dir = model_directory.as_os_str();

        assert_eq!(test_dir, expected_dir,);
    }
    #[test]
    fn test_file_path() {
        let model: Model = Model::default();
        let home_dir_string = std::env::var_os("HOME").unwrap();
        let home_path = std::path::Path::new(&home_dir_string);
        let mut model_directory = home_path.to_path_buf();
        // I don't actually know if this is correct.
        // TODO: support Windows
        #[cfg(target_os = "windows")]
        {
            model_directory.push("RoamingAppData");
            model_directory.push("WhisperGUI");
            model_directory.push("data");
            model_directory.push("models");
            model_directory.push("tiny.en.bin")
        }
        #[cfg(target_os = "macos")]
        {
            model_directory.push("Library");
            model_directory.push("Application Support");
            model_directory.push("com.jordan.WhisperGUI");
            model_directory.push("models");
            model_directory.push("tiny.en.bin")
        }
        #[cfg(target_os = "linux")]
        {
            model_directory.push(".local");
            model_directory.push("share");
            model_directory.push("whispergui");
            model_directory.push("models");
            model_directory.push("tiny.en.bin")
        }
        let test_file_path = model.file_path();
        let test_file = test_file_path.as_os_str();
        let expected_file = model_directory.as_os_str();

        assert_eq!(test_file, expected_file,);
    }

    #[test]
    fn test_url() {
        let model: Model = Model::default();

        let test_url = model.url();
        let test_url = test_url.as_str();

        assert_eq!(
            test_url,
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
        )
    }

    #[test]
    fn test_model_not_downloaded() {
        // Downloads the default model and checks to see that it's in the correct spot
        let model: Model = Model::default();
        assert!(!model.is_downloaded())
    }

    #[test]
    fn download_and_delete() {
        let model: Model = Model::default();
        model.download();
        assert!(model.is_downloaded());
        model.delete();
        assert!(!model.is_downloaded());
    }
}
