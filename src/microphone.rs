use cpal::traits::{DeviceTrait, HostTrait};

pub struct Microphone {
    host: cpal::Host,
    device: cpal::Device,
    pub sample_format: cpal::SampleFormat,
    pub stream: Result<cpal::Stream, cpal::BuildStreamError>,
    pub input_callback: Box<dyn Fn(&mut cpal::Data, &cpal::OutputCallbackInfo)>,
    pub error_callback: Box<dyn Fn(cpal::StreamError)>,
}

impl Microphone {
    pub fn init(&mut self) {
        self.host = cpal::default_host();
        self.device = self
            .host
            .default_input_device()
            .expect("no input device available");

        let mut supported_configs_range = self
            .device
            .supported_input_configs()
            .expect("error while querying configs");

        let supported_config = supported_configs_range
            .next()
            .expect("no supported configs")
            .with_max_sample_rate();

        let config = supported_config.into();

        self.sample_format = supported_config.sample_format();
        self.stream = self.device.build_input_stream(
            &config,
            move |data: &mut cpal::Data, info: &cpal::InputCallbackInfo| {
                self.input_callback(data, info)
            },
            move |err| self.error_callback(err),
            None,
        );
    }
}

// fn err(err: cpal::StreamError) {
//     eprintln!("an error occurred on the input audio stream: {}", err);
// }
