#![allow(clippy::uninlined_format_args)]

use std::thread::sleep;

use cpal::SampleFormat;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    WhisperToken,
};

use crate::microphone::Microphone;
use crate::model::Model;
use crate::preferences::Preferences;

const KEEP_MS: f64 = 200f64;
const SAMPLE_RATE: f64 = 16000f64;

const N_SAMPLES_KEEP: usize = ((1e-3 * KEEP_MS) * SAMPLE_RATE) as usize;

// TODO: possibly use a ring-buffer.
// TODO: Sliding-window
// TODO: transcriberstate -> possibly graph, or just booleans.
#[derive()]
struct Transcriber<'a, 'b> {
    input_buffer: Vec<u8>,
    // 32-bit buffer
    audio_buffer: Vec<f32>,
    output_buffer: Vec<String>,
    params: Option<FullParams<'a, 'b>>,
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

impl Transcriber {
    // TODO: refactor to take an optional microphone and handle accordingly?
    pub fn init(&mut self, prefs: Option<Preferences>) {
        if (self.initialized) {
            return;
        }
        self.microphone.input_callback =
            Box::<dyn Fn(&mut cpal::Data, &cpal::InputCallbackInfo)>::Box::new(
                move |data, info| self.push_input(data, info),
            );

        self.microphone.error_callback =
            Box::<dyn Fn(cpal::StreamError)>::Box::new(move |err| self.process_error(err));

        self.microphone.init();

        // Set params
        self.params = Option::from(FullParams::new(SamplingStrategy::Greedy { best_of: 1 }));
        self.ctx_params = Option::from(WhisperContextParameters::default());

        if prefs.is_none() {
            self.default_prefs();
        } else {
            self.set_prefs(prefs.as_ref().unwrap());
        }

        self.renew_ctx_and_state();

        self.initialized = true;
    }

    // Todo: refactor branch when GUI.
    fn run_model(&mut self) {
        // Check to see if model has been initialized.
        if !self.ready {
            return;
        }

        self.microphone.stream.play().unwrap();

        self.running = true;

        // TODO: threading + finished impl
        while self.running {
            // Get the time & check for pauses -> phrase-complete.
            let now = std::time::Instant::now();

            if !self.input_buffer.is_empty() {
                let mut phrase_complete = false;
                // TODO: look at libfvad for pauses
                // Or, time delta

                let state = self.whisper_state.as_mut().unwrap();

                // TODO: This needs a mutex.
                self.convert_audio();
                self.input_buffer.clear();

                // send the data to the model.
                state
                    .full(self.params.clone().unwrap(), self.audio_buffer.as_ref())
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

                let mut text = text.join("").trim();

                if phrase_complete {
                    self.output_buffer.push(String::from(text));
                } else {
                    self.output_buffer.last().expect("") = String::from(text);
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
                    let token_count = state.full_n_tokens(i);
                    for j in 0..token_count {
                        let token = state.full_get_token_id(i, j).expect("failed to get token");
                        tokens.push(token);
                    }
                }

                self.params
                    .as_mut()
                    .expect("params not set")
                    .set_tokens(tokens);

                // flush stdout.
                std::io::Write::stdout().flush().unwrap();
            } else {
                sleep(std::time::Duration::from_millis(100));
            }
        }
    }

    // Todo: refactor error handling.
    fn renew_ctx_and_state(&mut self) {
        // model should check to see if it's downloaded
        // if not, msg & prompt download & return.
        let model = self.model.as_ref().expect("Model not set");

        if !model.is_downloaded() {
            return;
        }

        self.ctx = Option::from(
            WhisperContext::new_with_params(
                model.file_path(),
                self.ctx_params.clone().expect("parameters not initialized"),
            )
            .expect("failed to load model"),
        );

        self.whisper_state = Option::from(
            self.ctx
                .as_ref()
                .unwrap()
                .create_state()
                .expect("failed to create state"),
        );

        self.ready = true;
    }

    fn push_input(&mut self, data: &mut cpal::Data, info: &cpal::InputCallbackInfo) {
        // Get the raw audio data
        self.input_buffer.extend_from_slice(data.bytes());
    }

    fn convert_audio(&mut self) {
        match self.microphone.sample_format() {
            SampleFormat::I8 => {
                self.convert_to_i16_sample(1);
            }
            SampleFormat::I16 => {
                self.convert_to_i16_sample(2);
            }
            SampleFormat::I32 => {
                self.convert_to_i16_sample(4);
            }
            SampleFormat::I64 => {
                self.convert_to_i16_sample(8);
            }
            SampleFormat::U8 => {
                self.convert_to_i16_sample(1);
            }
            SampleFormat::U16 => {
                self.convert_to_i16_sample(2);
            }
            SampleFormat::U32 => {
                self.convert_to_i16_sample(4);
            }
            SampleFormat::U64 => {
                self.convert_to_i16_sample(8);
            }
            SampleFormat::F32 => {
                let audio_data = self
                    .input_buffer
                    .clone()
                    .chunks_exact(4)
                    .into_iter()
                    .map(|n| f32::from_ne_bytes((0..4).map(|i| n[i]).collect().as_slice()))
                    .collect();
                let tmp_audio = whisper_rs::convert_stereo_to_mono_audio(&audio_data)
                    .expect("failed to convert to mono");
                self.audio_buffer.extend_from_slice(tmp_audio.as_slice());
            }
            SampleFormat::F64 => {
                let audio_data = self
                    .input_buffer
                    .clone()
                    .chunks_exact(8)
                    .into_iter()
                    .map(|n| f64::from_ne_bytes((0..8).map(|i| n[i]).collect().as_slice()) as f32)
                    .collect();
                let tmp_audio = whisper_rs::convert_stereo_to_mono_audio(&audio_data)
                    .expect("failed to convert to mono");
                self.audio_buffer.extend_from_slice(tmp_audio.as_slice());
            }
            _ => {}
        }
    }

    fn convert_to_i16_sample(&mut self, num_bytes: usize) {
        let mut inter_audio_data = Vec::with_capacity(self.input_buffer.len());
        let mapped_upcast: Vec<i16> = self
            .input_buffer
            .clone()
            .chunks_exact(num_bytes)
            .into_iter()
            .map(|n| i16::from_ne_bytes((0..num_bytes).map(|i| n[i]).collect()))
            .collect();
        whisper_rs::convert_integer_to_float_audio(mapped_upcast.as_slice(), &mut inter_audio_data)
            .expect("conversion failed");
        let tmp_audio = whisper_rs::convert_stereo_to_mono_audio(&inter_audio_data)
            .expect("failed to convert to mono");
        self.audio_buffer.extend_from_slice(tmp_audio.as_slice());
    }

    fn process_error(&mut self, err: cpal::StreamError) {
        // Pause the transcription
        self.microphone.stream.pause().unwrap();
        self.ready = false;

        // -> GUI should probably have some form of error.
        // For now, CLI.
        println!();
        eprintln!("an error occured on the output audio stream: {}", err);
    }

    fn set_prefs(&mut self, prefs: &Preferences) {
        self.params.set_n_threads(prefs.trans_prefs.n_threads);
        self.params.set_translate(prefs.trans_prefs.set_translate);
        self.params.set_language(prefs.trans_prefs.set_language);
        // Stdio only
        self.params
            .set_print_special(prefs.trans_prefs.print_special);
        self.params
            .set_print_progress(prefs.trans_prefs.print_progress);
        self.params
            .set_print_realtime(prefs.trans_prefs.print_realtime);
        self.params
            .set_print_timestamps(prefs.trans_prefs.print_timestamps);
        self.model = Option::from(prefs.trans_prefs.model.copy());
        self.ctx_params
            .as_mut()
            .expect("ctx params not set")
            .use_gpu = prefs.trans_prefs.use_gpu;
    }
    fn default_prefs(&mut self) {
        let min_thread = std::thread::available_parallelism()
            .expect("threading unavailable?")
            .unwrap() as std::ffi::c_int;
        self.params.set_n_threads(std::cmp::min(4, min_thread));
        self.params.set_translate(false);
        self.params.set_language(Some("en"));
        // Stdio only
        self.params.set_print_special(false);
        self.params.set_print_progress(false);
        self.params.set_print_realtime(false);
        self.params.set_print_timestamps(false);

        self.model = Model::default();
    }
}

impl Default for Transcriber {
    fn default() -> Self {
        Transcriber {
            input_buffer: vec![],
            // 16 khz, mono
            // TODO: figure out a default input buffer size.
            // Or perhaps allocate as needed
            audio_buffer: vec![0.0f32; 16000 * 2],
            // TODO: figure out a default output buffer size.
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
