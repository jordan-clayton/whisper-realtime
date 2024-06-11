#![allow(clippy::uninlined_format_args)]

use std::io;
use std::io::Write;
use std::thread::sleep;

use cpal::SampleFormat;
use cpal::traits::StreamTrait;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    WhisperToken,
};

use crate::microphone::Microphone;
use crate::model::Model;
use crate::preferences::Preferences;
use crate::ring_buffer::RingBuffer;
use crate::traits::{Queue, Transcriber};

const KEEP_MS: f64 = 200f64;
const SAMPLE_RATE: f64 = 16000f64;

const N_SAMPLES_KEEP: usize = ((1e-3 * KEEP_MS) * SAMPLE_RATE) as usize;

// TODO: Figure out a default input buffer size.
const INPUT_BUFFER_CAPACITY: usize = 10;

// TODO: Handle transcriber state.
// TODO: static audio processing

// TODO: token buffer.
#[derive()]
struct RealtimeTranscriber<'a> {
    input_buffer: RingBuffer<u8>,
    // 32-bit buffer
    audio_buffer: Vec<f32>,
    output_buffer: Vec<String>,
    params: Option<FullParams<'a, 'a>>,
    ctx: Option<WhisperContext>,
    ctx_params: Option<WhisperContextParameters>,
    whisper_state: Option<WhisperState<'a>>,
    model: Option<Model>,
    // This could also probably be an option.
    microphone: Microphone,
    initialized: bool,
    ready: bool,
    running: bool,
}

impl RealtimeTranscriber<'_> {
    pub fn new() -> Self {
        Self::default()
    }

    fn push_input(&mut self, data: &[u8], _info: &cpal::InputCallbackInfo) {
        for byte in data.iter() {
            self.input_buffer.push(byte);
        }
    }

    fn convert_to_i16_sample(&mut self, num_bytes: usize, buffer: &Vec<u8>) {
        let mut inter_audio_data = Vec::with_capacity(self.input_buffer.len());
        let mapped_upcast: Vec<i16> = buffer
            .clone()
            .chunks_exact(num_bytes)
            .into_iter()
            .map(|n| {
                let bytes: Vec<u8> = (0..num_bytes).map(|i| n[i]).collect();
                i16::from_ne_bytes(bytes.as_slice().try_into().unwrap())
            })
            .collect();
        whisper_rs::convert_integer_to_float_audio(mapped_upcast.as_slice(), &mut inter_audio_data)
            .expect("conversion failed");
        let tmp_audio = whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
            .expect("failed to convert to mono");
        self.audio_buffer.extend_from_slice(tmp_audio.as_slice());
    }

    fn process_stream_error(&mut self, err: cpal::StreamError) {
        // Pause the transcription
        self.microphone
            .stream
            .as_ref()
            .expect("stream not initialized")
            .pause()
            .unwrap();
        self.ready = false;

        // -> GUI should probably have some form of error.
        // For now, CLI.
        println!();
        eprintln!("an error occured on the output audio stream: {}", err);
    }

    pub fn empty_buffer(&mut self) -> Vec<u8> {
        let len = self.input_buffer.len();
        let mut audio: Vec<u8> = vec![0; len];

        for i in 0..len {
            audio[i] = self.input_buffer.pop();
        }

        audio
    }
}

impl<'a> Transcriber<'a> for RealtimeTranscriber<'a> {
    fn new() -> Self {
        RealtimeTranscriber::new()
    }

    // TODO: figure out how to handle this -> Both functions need to have static lifetimes.
    fn init(&mut self, prefs: Option<Preferences>) {
        if self.initialized {
            return;
        }

        let transcriber = std::sync::Arc::new(std::sync::Mutex::new(self));
        let input_callback = move |data: &[u8], info: &cpal::InputCallbackInfo| {
            let mut transcriber = transcriber.lock().unwrap();
            transcriber.push_input(data, info);
        };

        let output_callback = move |err: cpal::StreamError| {
            let mut transcriber = transcriber.lock().unwrap();
            transcriber.process_stream_error(err);
        };

        self.microphone.init(input_callback, output_callback);

        // Set params
        self.params = Some(FullParams::new(SamplingStrategy::Greedy { best_of: 1 }));
        self.ctx_params = Some(WhisperContextParameters::default());

        if prefs.is_none() {
            self.default_prefs();
        } else {
            self.set_prefs(prefs.as_ref().unwrap());
        }

        self.renew_ctx_and_state();

        self.initialized = true;
    }

    fn convert_audio(&mut self) {
        // Pop the current queue
        let audio: Vec<u8> = self.empty_buffer();

        match self.microphone.sample_format {
            SampleFormat::I8 => {
                self.convert_to_i16_sample(1, &audio);
            }
            SampleFormat::I16 => {
                self.convert_to_i16_sample(2, &audio);
            }
            SampleFormat::I32 => {
                self.convert_to_i16_sample(4, &audio);
            }
            SampleFormat::I64 => {
                self.convert_to_i16_sample(8, &audio);
            }
            SampleFormat::U8 => {
                self.convert_to_i16_sample(1, &audio);
            }
            SampleFormat::U16 => {
                self.convert_to_i16_sample(2, &audio);
            }
            SampleFormat::U32 => {
                self.convert_to_i16_sample(4, &audio);
            }
            SampleFormat::U64 => {
                self.convert_to_i16_sample(8, &audio);
            }
            SampleFormat::F32 => {
                let audio_data: Vec<f32> = audio
                    .clone()
                    .chunks_exact(4)
                    .into_iter()
                    .map(|n| f32::from_ne_bytes([n[0], n[1], n[2], n[3]]))
                    .collect();
                let tmp_audio = whisper_rs::convert_stereo_to_mono_audio(&audio_data)
                    .expect("failed to convert to mono");
                self.audio_buffer.extend_from_slice(tmp_audio.as_slice());
            }
            SampleFormat::F64 => {
                let audio_data: Vec<f32> = audio
                    .clone()
                    .chunks_exact(8)
                    .into_iter()
                    .map(|n| {
                        f64::from_ne_bytes([n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]]) as f32
                    })
                    .collect();
                let tmp_audio = whisper_rs::convert_stereo_to_mono_audio(&audio_data)
                    .expect("failed to convert to mono");
                self.audio_buffer.extend_from_slice(tmp_audio.as_slice());
            }
            _ => {}
        }
    }

    // Todo: refactor branch when GUI.
    fn process_audio(&mut self) -> String {
        // Check to see if model has been initialized.
        if !self.ready {
            return String::from("");
        }

        self.microphone
            .stream
            .as_ref()
            .expect("Stream not set up")
            .play()
            .unwrap();

        self.running = true;

        // TODO: threading + finished impl
        while self.running {
            // Get the time & check for pauses -> phrase-complete.
            let _now = std::time::Instant::now();

            if !self.input_buffer.is_empty() {
                let mut phrase_complete = false;
                // TODO: look at libfvad for pauses
                // Or, time delta

                // TODO: fix borrowing error
                let state = self.whisper_state.as_mut().expect("state not created");

                self.convert_audio();

                // TODO: fix double borrowing error
                state
                    .full(
                        *self.params.as_ref().expect("params not set").clone(),
                        self.audio_buffer.as_ref(),
                    )
                    .expect("failed to run model");

                let num_segments = state
                    .full_n_segments()
                    .expect("failed to get number of segments");

                let mut text: Vec<String> = vec![];

                for i in 0..num_segments {
                    let segment = state
                        .full_get_segment_text(i)
                        .expect("failed to get segment");

                    text.push(segment);
                }

                let text = text.join("");
                let text = text.trim();

                if phrase_complete {
                    self.output_buffer.push(String::from(text));
                } else {
                    // self.output_buffer.last().expect("") = String::from(text);
                    let last = self.output_buffer.len() - 1;
                    self.output_buffer[last] = String::from(text);
                }

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
                let keep_from = std::cmp::max(0, self.audio_buffer.len() - N_SAMPLES_KEEP - 1);
                self.audio_buffer = self.audio_buffer.drain(keep_from..).collect();

                // Seed the next prompt:
                // Note: if setting no_context, this needs to be skipped
                // TODO: handle Result properly
                let mut tokens: Vec<WhisperToken> = vec![];
                for i in 0..num_segments {
                    let token_count = state.full_n_tokens(i).expect("tokens failed");
                    for j in 0..token_count {
                        let token = state.full_get_token_id(i, j).expect("failed to get token");
                        tokens.push(token);
                    }
                }

                let c_tokens: Vec<std::ffi::c_int> = tokens
                    .into_iter()
                    .map(|token| token as std::ffi::c_int)
                    .collect();

                // TODO: figure out this lifetime error.
                self.params
                    .as_mut()
                    .expect("params not set")
                    .set_tokens(c_tokens.as_slice());

                // flush stdout.
                io::stdout().flush().unwrap();
            } else {
                sleep(std::time::Duration::from_millis(100));
            }
        }

        self.output_buffer.join("")
    }

    // Todo: refactor error handling.
    fn renew_ctx_and_state(&'a mut self) {
        // model should check to see if it's downloaded
        // if not, msg & prompt download & return.
        let model = self.model.as_ref().expect("Model not set");

        if !model.is_downloaded() {
            self.ready = false;
            return;
        }

        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu = self
            .ctx_params
            .as_ref()
            .expect("ctx params not initialized")
            .use_gpu;

        self.ctx = Some(
            WhisperContext::new_with_params(model.file_path(), ctx_params)
                .expect("model not found"),
        );

        self.whisper_state = Some(
            self.ctx
                .as_ref()
                .expect("no ctx")
                .create_state()
                .expect("failed to create state"),
        );
        self.ready = true;
    }

    fn set_prefs(&mut self, prefs: &Preferences) {
        let params = self.params.as_mut().expect("params not set");
        params.set_n_threads(prefs.trans_prefs.n_threads);
        params.set_translate(prefs.trans_prefs.set_translate);
        params.set_language(prefs.trans_prefs.set_language);
        // Stdio only
        params.set_print_special(prefs.trans_prefs.print_special);
        params.set_print_progress(prefs.trans_prefs.print_progress);
        params.set_print_realtime(prefs.trans_prefs.print_realtime);
        params.set_print_timestamps(prefs.trans_prefs.print_timestamps);
        self.model = Some(prefs.trans_prefs.model.clone());
        self.ctx_params
            .as_mut()
            .expect("ctx params not set")
            .use_gpu = prefs.trans_prefs.use_gpu;
    }

    fn default_prefs(&mut self) {
        let min_thread = std::thread::available_parallelism()
            .expect("threading unavailable?")
            .get() as std::ffi::c_int;

        let params = self.params.as_mut().expect("params not set");
        params.set_n_threads(std::cmp::min(4, min_thread));
        params.set_translate(false);
        params.set_language(Some("en"));
        // Stdio only
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        self.model = Some(Model::default());
    }
}

impl Default for RealtimeTranscriber<'_> {
    fn default() -> Self {
        RealtimeTranscriber {
            input_buffer: RingBuffer::with_capacity(INPUT_BUFFER_CAPACITY),
            audio_buffer: vec![],
            output_buffer: vec![],
            params: None,
            model: None,
            ctx: None,
            ctx_params: None,
            whisper_state: None,
            microphone: Microphone::new(),
            initialized: false,
            ready: false,
            running: false,
        }
    }
}
