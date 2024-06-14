#![allow(clippy::uninlined_format_args)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bbqueue_sync::{Consumer, Producer};
use cpal::SampleFormat;
use whisper_rs::{FullParams, SamplingStrategy, WhisperError, WhisperState, WhisperToken};

use crate::constants;
use crate::preferences::Configs;
use crate::traits::Transcriber;

// TODO: static audio processing

// Restructure:
/*
    configs,
    sample_format -> from mic-thread fn
    audio_buffer,
    output_buffer... possibly not all that necessary.
    input_buffer -> stream consumer from thread spawn fn.
    error_buffer -> stream error consumer from thread spawn fn.
    token_buffer -> fine as is.
    data_sender -> fine as is
    whisper_state -> fine as is
    ready -> from thread spawn fn
    running -> from thread spawn fn
*/

#[derive()]
pub struct RealtimeTranscriber<'a> {
    configs: Arc<Configs>,
    sample_format: SampleFormat,
    input_buffer: Arc<Consumer<'static, { constants::INPUT_BUFFER_CAPACITY }>>,
    input_error_buffer: Arc<Consumer<'static, { constants::CPAL_ERROR_BUFFER_CAPACITY }>>,
    // 32-bit buffer
    audio_buffer: Vec<f32>,
    // This might be wise to factor out.
    output_buffer: Vec<String>,
    // To send data to the G/UI
    data_sender: Arc<Producer<'static, { constants::OUTPUT_BUFFER_CAPACITY }>>,
    // To send errors to the G/UI
    error_sender: Arc<Producer<'static, { constants::ERROR_BUFFER_CAPACITY }>>,
    token_buffer: Vec<std::ffi::c_int>,
    // State is created before the transcriber is constructed.
    // The model is loaded into the ctx passed to the whisper state.
    whisper_state: WhisperState<'a>,
    // Transcriber state flags.
    // Ready -> selected Model is downloaded.
    ready: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
}

impl<'a> RealtimeTranscriber<'a> {
    pub fn new(
        state: WhisperState<'a>,
        sample_format: &SampleFormat,
        input_buffer: Arc<Consumer<'static, { constants::INPUT_BUFFER_CAPACITY }>>,
        input_error_buffer: Arc<Consumer<'static, { constants::CPAL_ERROR_BUFFER_CAPACITY }>>,
        data_sender: Arc<Producer<'static, { constants::OUTPUT_BUFFER_CAPACITY }>>,
        error_sender: Arc<Producer<'static, { constants::ERROR_BUFFER_CAPACITY }>>,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
    ) -> Self {
        let audio_buffer: Vec<f32> = vec![];
        let output_buffer: Vec<String> = vec![];
        let token_buffer: Vec<std::ffi::c_int> = vec![];
        let sample_format = *sample_format;

        RealtimeTranscriber {
            configs: Arc::new(Configs::default()),
            sample_format,
            input_buffer,
            input_error_buffer,
            audio_buffer,
            output_buffer,
            data_sender,
            error_sender,
            token_buffer,
            whisper_state: state,
            ready,
            running,
        }
    }

    pub fn new_with_configs(
        state: WhisperState<'a>,
        sample_format: &SampleFormat,
        input_buffer: Arc<Consumer<'static, { constants::INPUT_BUFFER_CAPACITY }>>,
        input_error_buffer: Arc<Consumer<'static, { constants::CPAL_ERROR_BUFFER_CAPACITY }>>,
        data_sender: Arc<Producer<'static, { constants::OUTPUT_BUFFER_CAPACITY }>>,
        error_sender: Arc<Producer<'static, { constants::ERROR_BUFFER_CAPACITY }>>,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
        configs: Arc<Configs>,
    ) -> Self {
        let audio_buffer: Vec<f32> = vec![];
        let output_buffer: Vec<String> = vec![];
        let token_buffer: Vec<std::ffi::c_int> = vec![];
        let sample_format = *sample_format;
        RealtimeTranscriber {
            configs,
            sample_format,
            input_buffer,
            input_error_buffer,
            audio_buffer,
            output_buffer,
            data_sender,
            error_sender,
            token_buffer,
            whisper_state: state,
            ready,
            running,
        }
    }

    fn send_data<const T: usize>(
        sender: &'a Producer<'static, T>,
        size_request: usize,
        bytes: &[u8],
    ) {
        let d_sender = sender.grant_max_remaining(size_request);
        if let Ok(mut d_gr) = d_sender {
            for byte in bytes.iter().enumerate() {
                let (i, b) = byte;
                d_gr[i] = *b;
            }
            d_gr.commit(size_request);
        }
    }
}

impl<'a: 'b, 'b> Transcriber<'a, 'b> for RealtimeTranscriber<'a> {
    fn process_audio(&'a mut self) -> String {
        // Check to see if mod has been initialized.
        let ready = self.ready.clone().load(Ordering::Relaxed);
        if !ready {
            return String::from("");
        }

        // This might be a borrow issue -> Possibly set & construct.
        // Just check here.
        let running = self.running.clone();
        running.store(true, Ordering::Relaxed);

        loop {
            let running = self.running.clone().load(Ordering::Relaxed);
            if !running {
                break;
            }
            // Get the time & check for pauses -> phrase-complete?
            let _now = std::time::Instant::now();

            let error_reader = &self.input_error_buffer;
            let e_reader = error_reader.read();

            // Check for mic input errors.
            if let Ok(mut g) = e_reader {
                // Bubble up the Stream error - another thread is responsible for stopping the transcription.
                let err = g.buf();
                let size_request = err.len();
                let e_sender = &self.error_sender;
                Self::send_data(e_sender, size_request, err);
                g.to_release(size_request);
            }

            let audio_reader = &self.input_buffer;
            let a_reader = audio_reader.read();
            match a_reader {
                Ok(mut grant) => {
                    let input_buffer = grant.buf();
                    let mut used_bytes = input_buffer.len();

                    if used_bytes % 2 == 1 {
                        used_bytes -= 1;
                    }

                    let mut inter_buffer: Vec<u8> = vec![0; used_bytes];

                    for i in 0..used_bytes {
                        inter_buffer[i] = input_buffer[i];
                    }

                    let input_buffer = inter_buffer.as_slice();

                    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                    let token_buffer = self.token_buffer.clone();
                    Self::set_full_params(&mut params, &self.configs, Some(&token_buffer));
                    let sample_format = &self.sample_format;
                    let sample_format = sample_format.clone();
                    let new_audio: Vec<f32> =
                        Self::convert_input_audio(input_buffer, sample_format);

                    self.audio_buffer.extend_from_slice(new_audio.as_slice());

                    let state = &mut self.whisper_state;

                    let result = state.full(params, &self.audio_buffer);

                    if let Err(e) = result {
                        match e {
                            WhisperError::NoSamples => {
                                println!("no samples");
                                continue;
                            }
                            // TODO: this should bubble the error up
                            // For now, exit the loop
                            _ => {
                                self.running.store(false, Ordering::Relaxed);
                                continue;
                            }
                        }
                    }

                    let num_segments = state.full_n_segments().expect("failed to get segments");
                    if num_segments == 0 {
                        continue;
                    }
                    let mut text: Vec<String> = vec![];

                    for i in 0..num_segments {
                        let segment = state
                            .full_get_segment_text(i)
                            .expect("failed to get segment");

                        // This needs to skip blank audio
                        text.push(segment);
                    }

                    let text = text.join("");
                    let text = text.trim();
                    let text_string = String::from(text);

                    self.output_buffer.push(text_string.clone());

                    // Send the new text to the G/UI.
                    let byte_string = text_string.into_bytes();
                    let size_request = byte_string.len();
                    let data_sender = &self.data_sender;
                    Self::send_data(data_sender, size_request, &byte_string);

                    // Keep a small amount of audio data for word boundaries.
                    // let keep_from =
                    //     std::cmp::max(0, self.audio_buffer.len() - constants::N_SAMPLES_KEEP - 1);
                    let keep_from = if self.audio_buffer.len() > (constants::N_SAMPLES_KEEP - 1) {
                        self.audio_buffer.len() - constants::N_SAMPLES_KEEP - 1
                    } else {
                        0
                    };
                    self.audio_buffer = self.audio_buffer.drain(keep_from..).collect();

                    // Seed the next prompt:
                    // Note: if setting no_context, this needs to be skipped
                    let mut tokens: Vec<WhisperToken> = vec![];
                    for i in 0..num_segments {
                        let token_count = state.full_n_tokens(i).expect("tokens failed");
                        for j in 0..token_count {
                            let token = state.full_get_token_id(i, j).expect("failed to get token");
                            tokens.push(token);
                        }
                    }

                    let new_tokens: Vec<std::ffi::c_int> = tokens
                        .into_iter()
                        .map(|token| token as std::ffi::c_int)
                        .collect();

                    self.token_buffer = new_tokens;

                    // Release the memory for writing.
                    grant.to_release(used_bytes);
                }
                Err(_e) => continue,
            }
        }

        self.output_buffer.join("").clone()
    }

    fn convert_input_audio(input_buffer: &[u8], sample_format: SampleFormat) -> Vec<f32> {
        let audio: Vec<u8> = Vec::from(input_buffer);
        let mut audio_data: Vec<f32> = vec![];

        let len = audio.len();
        let mut inter_audio_data: Vec<f32> = vec![0.0f32; len];
        match sample_format {
            SampleFormat::U8 => {
                let byte_vector: Vec<i16> = audio.clone().into_iter().map(|n| n as i16).collect();
                whisper_rs::convert_integer_to_float_audio(&byte_vector, &mut inter_audio_data)
                    .expect("conversion failed");
                audio_data = whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
                    .expect("failed to convert to mono");
            }
            SampleFormat::I16 => {
                let byte_vector: Vec<i16> = audio
                    .clone()
                    .chunks_exact(2)
                    .into_iter()
                    .map(|n| i16::from_ne_bytes([n[0], n[1]]))
                    .collect();
                whisper_rs::convert_integer_to_float_audio(&byte_vector, &mut inter_audio_data)
                    .expect("conversion failed");
                audio_data = whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
                    .expect("failed to convert to mono");
            }
            SampleFormat::F32 => {
                let inter_audio_data: Vec<f32> = audio
                    .clone()
                    .chunks_exact(4)
                    .into_iter()
                    .map(|n| f32::from_ne_bytes([n[0], n[1], n[2], n[3]]))
                    .collect();
                audio_data = whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
                    .expect("failed to convert to mono");
            }
            SampleFormat::F64 => {
                let inter_audio_data: Vec<f32> = audio
                    .clone()
                    .chunks_exact(8)
                    .into_iter()
                    .map(|n| {
                        f64::from_ne_bytes([n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]]) as f32
                    })
                    .collect();
                audio_data = whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
                    .expect("failed to convert to mono");
            }
            _ => {
                panic!("Unsupported format");
            }
        }

        audio_data
    }

    // fn convert_to_i16_sample_1_byte(buffer: &Vec<u8>, signed: bool) -> Vec<f32> {
    //     let mapped_cast: Vec<i16> = if signed {
    //         buffer
    //             .clone()
    //             .into_iter()
    //             .map(|n| {
    //                 let num = n as i8;
    //                 num as i16
    //             })
    //             .collect()
    //     } else {
    //         buffer.clone().into_iter().map(|n| n as i16).collect()
    //     };
    //
    //     let mapped_cast: Vec<i16> = buffer
    //         .clone()
    //         .into_iter()
    //         .map(|n| {
    //             let num = n as i8;
    //             num as i16
    //         })
    //         .collect();
    //
    //     let mut inter_audio_data = Vec::with_capacity(mapped_cast.len());
    //
    //     whisper_rs::convert_integer_to_float_audio(mapped_cast.as_slice(), &mut inter_audio_data)
    //         .expect("conversion failed");
    //     whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
    //         .expect("failed to convert to mono")
    // }
    // fn convert_to_i16_sample_2_byte(buffer: &Vec<u8>, signed: bool) -> Vec<f32> {
    //     let mapped_cast: Vec<i16> = if signed {
    //         buffer
    //             .clone()
    //             .chunks_exact(2)
    //             .into_iter()
    //             .map(|n| i16::from_ne_bytes([n[0], n[1]]))
    //             .collect()
    //     } else {
    //         buffer
    //             .clone()
    //             .chunks_exact(2)
    //             .into_iter()
    //             .map(|n| {
    //                 let num = u16::from_ne_bytes([n[0], n[1]]);
    //                 num as i16
    //             })
    //             .collect()
    //     };
    //
    //     let mut inter_audio_data = Vec::with_capacity(mapped_cast.len());
    //
    //     whisper_rs::convert_integer_to_float_audio(mapped_cast.as_slice(), &mut inter_audio_data)
    //         .expect("conversion failed");
    //     whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
    //         .expect("failed to convert to mono")
    // }
    // fn convert_to_i16_sample_4_byte(buffer: &Vec<u8>, signed: bool) -> Vec<f32> {
    //     let mapped_cast: Vec<i16> = if signed {
    //         buffer
    //             .clone()
    //             .chunks_exact(4)
    //             .into_iter()
    //             .map(|n| i32::from_ne_bytes([n[0], n[1], n[2], n[3]]) as i16)
    //             .collect()
    //     } else {
    //         buffer
    //             .clone()
    //             .chunks_exact(4)
    //             .into_iter()
    //             .map(|n| {
    //                 let num = u32::from_ne_bytes([n[0], n[1], n[2], n[3]]);
    //                 num as i16
    //             })
    //             .collect()
    //     };
    //
    //     let mut inter_audio_data = Vec::with_capacity(mapped_cast.len());
    //
    //     whisper_rs::convert_integer_to_float_audio(mapped_cast.as_slice(), &mut inter_audio_data)
    //         .expect("conversion failed");
    //     whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
    //         .expect("failed to convert to mono")
    // }
    // fn convert_to_i16_sample_8_byte(buffer: &Vec<u8>) -> Vec<f32> {
    //     let mapped_cast: Vec<i16> = buffer
    //         .clone()
    //         .chunks_exact(8)
    //         .into_iter()
    //         .map(|n| i16::from_ne_bytes([n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]]))
    //         .collect();
    //
    //     let mut inter_audio_data = Vec::with_capacity(mapped_cast.len());
    //
    //     whisper_rs::convert_integer_to_float_audio(mapped_cast.as_slice(), &mut inter_audio_data)
    //         .expect("conversion failed");
    //     whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
    //         .expect("failed to convert to mono")
    // }

    // fn convert_to_i16_sample(byte_chunks: usize, buffer: &Vec<u8>) -> Vec<f32> {
    //     let mapped_cast: Vec<i16> = buffer
    //         .clone()
    //         .chunks_exact(byte_chunks)
    //         .into_iter()
    //         .map(|n| {
    //             let bytes: Vec<u8> = (0..byte_chunks).map(|i| n[i]).collect();
    //             println!("bytes len: {}", bytes.len());
    //
    //             i16::from_ne_bytes(bytes.as_slice().try_into().unwrap())
    //         })
    //         .collect();
    //
    //     let mut inter_audio_data = Vec::with_capacity(mapped_cast.len());
    //
    //     whisper_rs::convert_integer_to_float_audio(mapped_cast.as_slice(), &mut inter_audio_data)
    //         .expect("conversion failed");
    //     whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
    //         .expect("failed to convert to mono")
    // }

    // TODO: refactor these?
    fn set_full_params(
        full_params: &mut FullParams<'b, 'b>,
        prefs: &'a Configs,
        tokens: Option<&'b Vec<std::ffi::c_int>>,
    ) {
        let lang = prefs.set_language.as_ref();

        full_params.set_n_threads(prefs.n_threads);
        full_params.set_n_threads(prefs.n_threads);
        full_params.set_translate(prefs.set_translate);

        if lang.is_some() {
            full_params.set_language(Some(lang.unwrap().as_str()));
        } else {
            full_params.set_language(Some("auto"))
        }

        // // Stdio only
        full_params.set_print_special(prefs.print_special);
        full_params.set_print_progress(prefs.print_progress);
        full_params.set_print_realtime(prefs.print_realtime);
        full_params.set_print_timestamps(prefs.print_timestamps);

        if tokens.is_some() {
            let token_buffer = tokens.unwrap();
            full_params.set_tokens(&token_buffer)
        }
    }
}
