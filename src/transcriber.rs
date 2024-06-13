#![allow(clippy::uninlined_format_args)]

use std::io::{stdout, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bbqueue_sync::Producer;
use cpal::SampleFormat;
use whisper_rs::{FullParams, SamplingStrategy, WhisperState, WhisperToken};

use crate::constants;
use crate::microphone::Microphone;
use crate::preferences::Configs;
use crate::traits::Transcriber;

// TODO: static audio processing

#[derive()]
struct RealtimeTranscriber<'a> {
    configs: Configs,
    // 32-bit buffer
    audio_buffer: Vec<f32>,
    // This might be wise to factor out.
    output_buffer: Vec<String>,
    // To send data to the G/UI
    data_sender: Producer<'static, { constants::OUTPUT_BUFFER_CAPACITY }>,
    // To send errors to the G/UI
    error_sender: Producer<'static, { constants::ERROR_BUFFER_CAPACITY }>,
    token_buffer: Vec<std::ffi::c_int>,
    // State is created before the transcriber is constructed.
    // The model is loaded into the ctx passed to the whisper state.
    whisper_state: WhisperState<'a>,
    // Microphone has stream, sample format, input buffer and error buffer
    microphone: Microphone<'a>,
    // Transcriber state flags.
    // Ready -> selected Model is downloaded.
    ready: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
}

impl<'a> RealtimeTranscriber<'a> {
    pub fn new(
        state: WhisperState<'a>,
        microphone: Microphone<'a>,
        data_sender: Producer<'static, { constants::OUTPUT_BUFFER_CAPACITY }>,
        error_sender: Producer<'static, { constants::ERROR_BUFFER_CAPACITY }>,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
    ) -> Self {
        let audio_buffer: Vec<f32> = vec![];
        let output_buffer: Vec<String> = vec![];
        let token_buffer: Vec<std::ffi::c_int> = vec![];

        RealtimeTranscriber {
            configs: Configs::default(),
            audio_buffer,
            output_buffer,
            data_sender,
            error_sender,
            token_buffer,
            whisper_state: state,
            microphone,
            ready,
            running,
        }
    }

    fn new_with_configs(
        state: WhisperState<'a>,
        microphone: Microphone<'a>,
        data_sender: Producer<'static, { constants::OUTPUT_BUFFER_CAPACITY }>,
        error_sender: Producer<'static, { constants::ERROR_BUFFER_CAPACITY }>,
        running: Arc<AtomicBool>,
        ready: Arc<AtomicBool>,
        configs: Configs,
    ) -> Self {
        let audio_buffer: Vec<f32> = vec![];
        let output_buffer: Vec<String> = vec![];
        let token_buffer: Vec<std::ffi::c_int> = vec![];
        RealtimeTranscriber {
            configs,
            audio_buffer,
            output_buffer,
            data_sender,
            error_sender,
            token_buffer,
            whisper_state: state,
            microphone,
            ready,
            running,
        }
    }
}

impl<'a: 'b, 'b> Transcriber<'a, 'b> for RealtimeTranscriber<'a> {
    // This struct needs to be able to be ARC'd onto another thread.
    // Structure:
    // Check atomic.running
    // Consume the audio error queue -> if it has data, break the loop and return the... error?
    // Consume the audio data queue -> if it has data, process, else (insuf size for read) uh, continue.
    // Process audio -> run the model -> (this is mostly okay).
    // On finish, drop the stream & return the string.
    // (this can happen in thread caller - closer to UI) -> drop the microphone, join the prod/cons back into buffer & re-split
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

            let mic = &self.microphone;
            let reader = &mic.input_buffer;
            let reader = reader.read();
            match reader {
                Ok(grant) => {
                    // TODO: look at libfvad for pauses
                    let input_buffer = grant.buf();
                    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                    let token_buffer = self.token_buffer.clone();
                    Self::set_full_params(&mut params, &self.configs, Some(&token_buffer));
                    let new_audio: Vec<f32> =
                        Self::convert_input_audio(input_buffer, mic.sample_format.clone());
                    self.audio_buffer.extend_from_slice(new_audio.as_slice());

                    let state = &mut self.whisper_state;

                    state
                        .full(params, &self.audio_buffer)
                        .expect("model failed");

                    let num_segments = state.full_n_segments().expect("failed to get segments");
                    let mut text: Vec<String> = vec![];

                    for i in 0..num_segments {
                        let segment = state
                            .full_get_segment_text(i)
                            .expect("failed to get segment");

                        text.push(segment);
                    }

                    let text = text.join("");
                    let text = text.trim();

                    self.output_buffer.push(String::from(text));

                    // TODO: implement gui
                    // -> clear stdout (linux/mac only)
                    std::process::Command::new("clear")
                        .status()
                        .expect("failed to clear screen");

                    // Windows..?
                    // std::process::Command::new("cls")
                    //     .status()
                    //     .expect("failed to clear screen");

                    // Handle this in the gui, but for now, print to stdout.
                    for line in self.output_buffer.iter() {
                        print!("{}", line);
                    }

                    // Keep a small amount of audio data for word boundaries.
                    let keep_from =
                        std::cmp::max(0, self.audio_buffer.len() - constants::N_SAMPLES_KEEP - 1);
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

                    // flush stdout.
                    stdout().flush().unwrap();
                }
                Err(_e) => continue,
            }
        }

        self.output_buffer.join("")
    }

    fn convert_input_audio(input_buffer: &[u8], sample_format: SampleFormat) -> Vec<f32> {
        let audio: Vec<u8> = Vec::from(input_buffer);
        let mut audio_data: Vec<f32> = vec![];

        match sample_format {
            SampleFormat::I8 => {
                audio_data = Self::convert_to_i16_sample(1, &audio);
            }
            SampleFormat::I16 => {
                audio_data = Self::convert_to_i16_sample(2, &audio);
            }
            SampleFormat::I32 => {
                audio_data = Self::convert_to_i16_sample(4, &audio);
            }
            SampleFormat::I64 => {
                audio_data = Self::convert_to_i16_sample(8, &audio);
            }
            SampleFormat::U8 => {
                audio_data = Self::convert_to_i16_sample(1, &audio);
            }
            SampleFormat::U16 => {
                audio_data = Self::convert_to_i16_sample(2, &audio);
            }
            SampleFormat::U32 => {
                audio_data = Self::convert_to_i16_sample(4, &audio);
            }
            SampleFormat::U64 => {
                audio_data = Self::convert_to_i16_sample(8, &audio);
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
            _ => {}
        }
        audio_data
    }

    fn convert_to_i16_sample(byte_chunks: usize, buffer: &Vec<u8>) -> Vec<f32> {
        let mapped_cast: Vec<i16> = buffer
            .clone()
            .chunks_exact(byte_chunks)
            .into_iter()
            .map(|n| {
                let bytes: Vec<u8> = (0..byte_chunks).map(|i| n[i]).collect();
                i16::from_ne_bytes(bytes.as_slice().try_into().unwrap())
            })
            .collect();

        let mut inter_audio_data = Vec::with_capacity(mapped_cast.len());

        whisper_rs::convert_integer_to_float_audio(mapped_cast.as_slice(), &mut inter_audio_data)
            .expect("conversion failed");
        whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
            .expect("failed to convert to mono")
    }

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
