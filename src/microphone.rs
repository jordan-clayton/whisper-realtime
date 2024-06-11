use cpal::traits::{DeviceTrait, HostTrait};

pub struct Microphone {
    host: cpal::Host,
    device: Option<cpal::Device>,
    pub sample_format: cpal::SampleFormat,
    pub stream: Option<cpal::Stream>,
}

impl Microphone {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn init<F, E>(&mut self, input_callback: F, error_callback: E)
    where
        F: FnMut(&[u8], &cpal::InputCallbackInfo) + Send + 'static,
        E: FnMut(cpal::StreamError) + Send + 'static,
    {
        let mut supported_configs_range = self
            .device
            .as_ref()
            .expect("no input device available")
            .supported_input_configs()
            .expect("error while querying configs");

        let supported_config = supported_configs_range
            .next()
            .expect("no supported configs")
            .with_max_sample_rate();

        let config = supported_config.clone().into();

        self.sample_format = supported_config.sample_format();

        self.stream = Option::from(
            self.device
                .as_ref()
                .unwrap()
                .build_input_stream(&config, input_callback, error_callback, None)
                .expect("Failed to build audio stream"),
        );
    }
}

impl Default for Microphone {
    fn default() -> Self {
        Microphone {
            host: cpal::default_host(),
            device: cpal::default_host().default_input_device(),
            sample_format: cpal::SampleFormat::I8,
            stream: None,
        }
    }
}
