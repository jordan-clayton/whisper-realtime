use std::sync::Arc;

use bbqueue_sync::Producer;
use cpal::{Sample, SampleFormat, StreamError};
use cpal::traits::{DeviceTrait, HostTrait};

use crate::constants;

// pub struct Microphone {
//     pub sample_format: cpal::SampleFormat,
//     pub stream: cpal::Stream,
//     pub input_buffer: Consumer<'static, { constants::INPUT_BUFFER_CAPACITY }>,
//     pub error_buffer: Consumer<'static, { constants::CPAL_ERROR_BUFFER_CAPACITY }>,
// }

// TODO: Refactor to use ringbuffer

pub fn create_microphone_stream(
    mic_producer: Arc<Producer<'static, { constants::INPUT_BUFFER_CAPACITY }>>,
    error_producer: Arc<Producer<'static, { constants::CPAL_ERROR_BUFFER_CAPACITY }>>,
    device_host: Option<cpal::Host>,
    input_device_name: Option<&String>,
) -> (cpal::Stream, SampleFormat) {
    let host: cpal::Host = if device_host.is_some() {
        device_host.unwrap()
    } else {
        cpal::default_host()
    };

    // -> Check for stored device by name if list is available, otherwise default
    let device = if input_device_name.is_some() {
        let name = input_device_name.unwrap().clone();
        let device_list = host.input_devices();
        if let Ok(mut devices) = device_list {
            let device = devices.find(|d| d.name().map(|n| n == name).unwrap_or(false));
            if device.is_some() {
                device
            } else {
                println!("Did not find {}", &input_device_name.unwrap());
                host.default_input_device()
            }
        } else {
            host.default_input_device()
        }
    } else {
        host.default_input_device()
    }
    .expect("failed to find input device");

    println!("device: {}", &device.name().unwrap());

    let mut supported_configs_range = device
        .supported_input_configs()
        .expect("config query error");
    let supported_config = supported_configs_range
        .next()
        .expect("no supported configs")
        .with_max_sample_rate();
    let mut config: cpal::StreamConfig = supported_config.clone().into();
    // TODO: This does not need to be fixed
    let buffer_size = cpal::BufferSize::Fixed(4096);

    // TODO: SAMPLE_RATE should probably be the same as WHISPER.
    config.sample_rate = cpal::SampleRate(48000);
    config.buffer_size = buffer_size;
    // TODO: this needs to be f32.
    let sample_format = supported_config.sample_format();
    println!("Sample format: {}", sample_format);

    let d_send = mic_producer.clone();
    let e_send = error_producer.clone();

    // TODO: figure out some way to generalize the repeated code ->
    // TODO: look at libfvad for pauses - blank audio is overflowing the buffer
    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _info| {
                let size_request = data.len();
                let grant = d_send.grant_max_remaining(size_request);
                match grant {
                    Ok(mut g) => {
                        // Convert into ne bytes.
                        let mut data_array: Vec<f32> = vec![];
                        data_array.extend_from_slice(data);

                        for data in data_array {
                            let data_bytes = data.to_ne_bytes();
                            for byte in data_bytes.iter().enumerate() {
                                let (i, b) = byte;
                                g[i] = *b;
                            }
                        }
                        g.commit(size_request);
                    }
                    // Zero space... not quite sure what to do with this yet
                    Err(_e) => {}
                }
            },
            move |err| send_error(&e_send, err),
            None,
        ),
        SampleFormat::F64 => device.build_input_stream(
            &config,
            move |data: &[f64], _info| {
                let size_request = data.len();
                let grant = d_send.grant_max_remaining(size_request);
                match grant {
                    Ok(mut g) => {
                        // Convert into ne bytes.
                        let mut data_array: Vec<f64> = vec![];
                        data_array.extend_from_slice(data);
                        for data in data_array {
                            let data_bytes = data.to_ne_bytes();
                            for byte in data_bytes.iter().enumerate() {
                                let (i, b) = byte;
                                g[i] = *b;
                            }
                        }
                        g.commit(size_request);
                    }
                    // Zero space... not quite sure what to do with this yet
                    Err(_e) => {}
                }
            },
            move |err| send_error(&e_send, err),
            None,
        ),
        // TODO: this needs to be the only sample rate -> for libvfad pause detection
        // send in 80, 160, or 240 length chunks -> if, and only if there is no pause.
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _info| {
                let size_request = data.len();
                let grant = d_send.grant_max_remaining(size_request);
                match grant {
                    Ok(mut g) => {
                        // Convert into ne bytes.
                        let mut data_array: Vec<i16> = vec![];
                        data_array.extend_from_slice(data);
                        for data in data_array {
                            let data_bytes = data.to_ne_bytes();
                            for byte in data_bytes.iter().enumerate() {
                                let (i, b) = byte;
                                g[i] = *b;
                            }
                        }
                        g.commit(size_request);
                    }
                    // Zero space... not quite sure what to do with this yet
                    Err(_e) => {}
                }
            },
            move |err| send_error(&e_send, err),
            None,
        ),
        SampleFormat::U8 => device.build_input_stream(
            &config,
            move |data: &[u8], _info| {
                let size_request = data.len();
                let grant = d_send.grant_max_remaining(size_request);
                match grant {
                    Ok(mut g) => {
                        let len = g.buf().len();

                        for i in 0..len {
                            g[i] = data[i];
                        }
                        // Convert into ne bytes.
                        g.commit(size_request);
                    }
                    // Zero space... not quite sure what to do with this yet
                    Err(_e) => {}
                }
            },
            move |err| send_error(&e_send, err),
            None,
        ),
        sample_format => panic!("Unsupported sample format: {sample_format}"),
    }
    .expect("failed to build input stream");

    // let stream = device
    //     .build_input_stream(
    //         &config,
    //         move |data: &[u8], _info| {
    //             let size_request = data.len();
    //             let grant = d_send.grant_max_remaining(size_request);
    //             match grant {
    //                 Ok(mut g) => {
    //                     // Copy bytes over to buffer.
    //                     for element in data.iter().enumerate() {
    //                         let (i, byte) = element;
    //                         g[i] = *byte;
    //                     }
    //                     g.commit(size_request);
    //                 }
    //                 // Zero space... not quite sure what to do with this yet
    //                 Err(_e) => {}
    //             }
    //         },
    //         // Not quite sure how to do this. Might be better to use a mpsc queue
    //         move |err| {
    //             let size_request = std::mem::size_of::<StreamError>();
    //             let grant = e_send.grant_max_remaining(size_request);
    //             match grant {
    //                 // TODO: maybe use serde...
    //                 Ok(mut g) => unsafe {
    //                     let buf = g.buf();
    //                     let e_ptr = buf.as_mut_ptr();
    //                     let e_ptr = e_ptr as *mut StreamError;
    //                     *e_ptr = err;
    //                     g.commit(size_request);
    //                 },
    //                 // Zero space... not quite sure what to do with this yet
    //                 Err(_e) => {}
    //             }
    //         },
    //         None,
    //     )
    //     .expect("failed to build audio stream");
    (stream, sample_format)
}

// This won't work
fn send_data<T: Sample>(
    data: &mut &[T],
    data_producer: Arc<Producer<'static, { constants::INPUT_BUFFER_CAPACITY }>>,
) {
}

fn send_error(
    error_producer: &Arc<Producer<'static, { constants::CPAL_ERROR_BUFFER_CAPACITY }>>,
    err: StreamError,
) {
    let size_request = std::mem::size_of::<StreamError>();
    let grant = error_producer.grant_max_remaining(size_request);
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
}
