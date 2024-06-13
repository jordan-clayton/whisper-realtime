use std::sync::Arc;

use bbqueue_sync::{BBBuffer, Consumer};
use cpal::StreamError;
use cpal::traits::{DeviceTrait, HostTrait};

use crate::constants;

pub struct Microphone<'a> {
    pub sample_format: cpal::SampleFormat,
    pub stream: cpal::Stream,
    pub input_buffer: Consumer<'a, { constants::INPUT_BUFFER_CAPACITY }>,
    pub error_buffer: Consumer<'a, { constants::CPAL_ERROR_BUFFER_CAPACITY }>,
}

// TODO: eventually add input configs.
impl Microphone<'_> {
    pub fn new(
        mic_data_buffer: &'static BBBuffer<{ constants::INPUT_BUFFER_CAPACITY }>,
        mic_error_buffer: &'static BBBuffer<{ constants::CPAL_ERROR_BUFFER_CAPACITY }>,
        device_host: Option<cpal::Host>,
        input_device: Option<String>,
    ) -> Self {
        let host = if device_host.is_some() {
            device_host.unwrap()
        } else {
            cpal::default_host()
        };

        // -> Check for stored device by name if list is available, otherwise default
        let device = if input_device.is_some() {
            let name = input_device.unwrap();
            let device_list = host.input_devices();
            if let Ok(mut devices) = device_list {
                let device = devices.find(|d| d.name().map(|n| n == name).unwrap_or(false));
                if device.is_some() {
                    device
                } else {
                    host.default_input_device()
                }
            } else {
                host.default_input_device()
            }
        } else {
            host.default_input_device()
        }
        .expect("failed to find input device");

        let mut supported_configs_range = device
            .supported_input_configs()
            .expect("config query error");
        let supported_config = supported_configs_range
            .next()
            .expect("no supported configs")
            .with_max_sample_rate();
        let config = supported_config.clone().into();
        let sample_format = supported_config.sample_format();

        let (d_prod, d_cons) = mic_data_buffer
            .try_split()
            .expect("unable to create audio buffer");
        let (e_prod, e_cons) = mic_error_buffer
            .try_split()
            .expect("unable to create error buffer");

        let d_send = Arc::new(d_prod);
        let e_send = Arc::new(e_prod);
        let c_d_send = d_send.clone();
        let c_e_send = e_send.clone();

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[u8], _info| {
                    let size_request = data.len();
                    let grant = c_d_send.grant_max_remaining(size_request);
                    match grant {
                        Ok(mut g) => {
                            // Copy bytes over to buffer.
                            for element in data.iter().enumerate() {
                                let (i, byte) = element;
                                g[i] = *byte;
                            }
                            g.commit(size_request);
                        }
                        // Zero space... not quite sure what to do with this yet
                        Err(_e) => {}
                    }
                },
                // Not quite sure how to do this. Might be better to use a mpsc queue
                move |err| {
                    let size_request = std::mem::size_of::<StreamError>();
                    let grant = c_e_send.grant_max_remaining(size_request);
                    match grant {
                        // TODO: maybe use serde...
                        Ok(mut g) => unsafe {
                            let buf = g.buf();
                            let e_ptr = buf.as_mut_ptr();
                            let e_ptr = e_ptr as *mut StreamError;
                            *e_ptr = err;
                            g.commit(size_request);
                        },
                        // Zero space... not quite sure what to do with this yet
                        Err(_e) => {}
                    }
                },
                None,
            )
            .expect("failed to build audio stream");

        Microphone {
            sample_format: sample_format.to_owned(),
            stream,
            input_buffer: d_cons,
            error_buffer: e_cons,
        }
    }
}
