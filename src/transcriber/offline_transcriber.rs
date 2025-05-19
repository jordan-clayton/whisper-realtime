use std::ffi::c_void;
use std::sync::{Arc, LazyLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::transcriber::transcriber::{
    CallbackTranscriber, Transcriber, WhisperCallbacks, WhisperOutput, WhisperProgressCallback,
};
use crate::transcriber::vad::VAD;
use crate::utils::errors::WhisperRealtimeError;
use crate::utils::sender::Sender;
use crate::whisper::configs::WhisperConfigsV2;

/// NOTE: This will no longer compile. The transcriber trait has changed and this struct requires
/// re-implementation. Refactoring is still in progress, but expect this to be fixed very soon.

// Workaround for whisper-rs issue #134 -- moving memory in rust causes a segmentation fault
// in the progress callback.
// Solution from: https://github.com/thewh1teagle/vibe/blob/main/core/src/transcribe.rs
// Until this bug is fixed, there is only one single callback point for a progress callback.
// Thus it is not advised to run multiple OfflineTranscribers across threads with callbacks.
// TODO: Remove this workaround if the safe functions are calling properly.
static PROGRESS_CALLBACK: LazyLock<
    Mutex<Option<Box<dyn WhisperProgressCallback<Argument = i32>>>>,
> = LazyLock::new(|| Mutex::new(None));

// TODO: Migrate this to the audio module; it doesn't belong here
// TODO: look into using slices over vectors
pub enum SupportedAudioSample {
    I16(Vec<i16>),
    F32(Vec<f32>),
}

// TODO: Migrate this to the audio module; it doesn't belong here
// TODO: rename this to something clearer.
#[derive(Copy, Clone, PartialEq)]
pub enum SupportedChannels {
    Mono,
    Stereo,
}

// TODO: proper documentation
// Note: since audio is implicitly converted to a format understandable by whisper,
// it isn't necessary for the VAD to be generic.
#[derive(Clone)]
pub struct OfflineTranscriberBuilder<V>
where
    V: VAD<f32>,
{
    configs: Option<Arc<WhisperConfigsV2>>,
    sender: Option<Sender<WhisperOutput>>,
    audio: Option<Arc<SupportedAudioSample>>,
    channels: Option<SupportedChannels>,
    voice_activity_detector: Option<Arc<Mutex<V>>>,
}

impl<V: VAD<f32>> OfflineTranscriberBuilder<V> {
    pub fn new() -> Self {
        Self {
            configs: None,
            sender: None,
            audio: None,
            channels: None,
            voice_activity_detector: None,
        }
    }
    pub fn with_configs(mut self, configs: WhisperConfigsV2) -> Self {
        self.configs = Some(Arc::new(configs));
        self
    }
    pub fn with_sender(mut self, sender: Sender<WhisperOutput>) -> Self {
        self.sender = Some(sender);
        self
    }
    pub fn with_audio(mut self, audio: SupportedAudioSample) -> Self {
        self.audio = Some(Arc::new(audio));
        self
    }
    pub fn with_channel_configurations(mut self, channels: SupportedChannels) -> Self {
        self.channels = Some(channels);
        self
    }
    pub fn with_voice_activity_detector<V2: VAD<f32>>(
        self,
        vad: V2,
    ) -> OfflineTranscriberBuilder<V2> {
        let v = Arc::new(Mutex::new(vad));
        OfflineTranscriberBuilder {
            configs: self.configs,
            sender: self.sender,
            audio: self.audio,
            channels: self.channels,
            voice_activity_detector: Some(v),
        }
    }
    pub fn build(self) -> Result<OfflineTranscriber<V>, WhisperRealtimeError> {
        let configs = self.configs.ok_or(WhisperRealtimeError::ParameterError(
            "Configs missing in OfflineTranscriberBuilder..".to_string(),
        ))?;
        let sender = self.sender.ok_or(WhisperRealtimeError::ParameterError(
            "Output sender missing in OfflineTranscriberBuilder.".to_string(),
        ))?;
        let audio = self.audio.ok_or(WhisperRealtimeError::ParameterError(
            "Audio missing in OfflineTranscriberBuilder.".to_string(),
        ))?;
        let channels = self.channels.ok_or(WhisperRealtimeError::ParameterError(
            "Channel configurations missing in OfflineTranscriberBuilder.".to_string(),
        ))?;
        // Vad can be None; if there is no VAD provided, the full speech will be processed.
        let vad = self.voice_activity_detector;
        Ok(OfflineTranscriber {
            configs,
            sender,
            audio,
            channels,
            voice_activity_detector: vad,
        })
    }
}

#[derive(Clone)]
pub struct OfflineTranscriber<V: VAD<f32>> {
    configs: Arc<WhisperConfigsV2>,
    sender: Sender<WhisperOutput>,
    audio: Arc<SupportedAudioSample>,
    // Supported Channels is just an enumeration member and does not need to be shared.
    channels: SupportedChannels,
    voice_activity_detector: Option<Arc<Mutex<V>>>,
}

impl<V: VAD<f32>> OfflineTranscriber<V> {
    fn run_transcription(
        &mut self,
        full_params: whisper_rs::FullParams,
    ) -> Result<String, WhisperRealtimeError> {
        let whisper_context_params = self.configs.to_whisper_context_params();
        // Set up a whisper context
        let ctx = whisper_rs::WhisperContext::new_with_params(
            self.configs
                .model()
                .file_path()
                .to_str()
                .expect("File should be a valid utf-8 str"),
            whisper_context_params,
        )?;

        let mut whisper_state = ctx.create_state()?;

        // Prepare audio
        let mut audio_samples = match self.audio.as_ref() {
            SupportedAudioSample::I16(audio) => {
                let len = audio.len();
                let mut float_samples = vec![0.0; len];
                whisper_rs::convert_integer_to_float_audio(audio, &mut float_samples)?;
                float_samples
            }
            SupportedAudioSample::F32(audio) => audio.clone(),
        };

        // Extract speech frames if there's a VAD
        if let Some(try_vad) = self.voice_activity_detector.as_ref() {
            let mut vad = match try_vad.lock() {
                Ok(vad) => vad,
                Err(e) => {
                    try_vad.clear_poison();
                    e.into_inner()
                }
            };
            audio_samples = vad.extract_voiced_frames(&audio_samples);
        }

        let mono_audio = match self.channels {
            SupportedChannels::Mono => audio_samples,
            SupportedChannels::Stereo => whisper_rs::convert_stereo_to_mono_audio(&audio_samples)?,
        };

        let _ = whisper_state.full(full_params, &mono_audio)?;
        // I'm not quite sure what the error conditions are for this to fail
        let num_segments = whisper_state.full_n_segments()?;
        let mut text: Vec<String> = vec![];
        text.reserve(num_segments as usize);

        // Transcribe segments and send through a channel to a UI.
        for i in 0..num_segments {
            if let Ok(segment) = whisper_state.full_get_segment_text(i) {
                text.push(segment.clone());
                // This function doesn't need to stop if the channel is closed, so just ignore.
                let _ = self.sender.send(WhisperOutput::FinishedPhrase(segment));
            }
        }

        // Return the final transcription
        // Offline segments are generally longer phrases and should be separated by a newline.
        Ok(text.join("\n"))
    }
}

impl<V: VAD<f32>> Transcriber for OfflineTranscriber<V> {
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, WhisperRealtimeError> {
        let confs = Arc::clone(&self.configs);

        let mut full_params = confs.to_whisper_full_params();
        // Set the abort callback -> check and make sure this no longer needs to touch the
        // legacy extern C callback.
        let rt = Arc::clone(&run_transcription);

        // This might segfault, as per the following: https://github.com/tazz4843/whisper-rs/issues/114
        // I'm not sure as to why that might be at the moment; if so, move to the unsafe C api
        full_params.set_abort_callback_safe(move || rt.load(Ordering::Acquire));
        self.run_transcription(full_params)
    }
}

impl<V, P> CallbackTranscriber<P> for OfflineTranscriber<V>
where
    V: VAD<f32>,
    P: WhisperProgressCallback,
{
    fn process_with_callbacks(
        &mut self,
        run_transcription: Arc<AtomicBool>,
        callbacks: WhisperCallbacks<P>,
    ) -> Result<String, WhisperRealtimeError> {
        // Decompose the callbacks struct
        let WhisperCallbacks {
            progress: maybe_progress_callback,
        } = callbacks;

        let confs = Arc::clone(&self.configs);
        let mut full_params = confs.to_whisper_full_params();
        // This might segfault, for the same reasons mentioned above in the Transcriber impl
        // If so, migrate to the unsafe C API
        if let Some(mut callback) = maybe_progress_callback {
            // Set the progress callback
            full_params.set_progress_callback_safe(move |p| {
                callback.call(p);
            });
        }
        let rt = Arc::clone(&run_transcription);

        full_params.set_abort_callback_safe(move || rt.load(Ordering::Acquire));
        self.run_transcription(full_params)
    }
}

/// This small function runs each time the encoder fires up to check the running state of the
/// transcriber. Allows a user to deliberately early terminate transcription instead of waiting
/// for the full audio to be transcribed.

// TODO: remove this if the safe callbacks no longer segfault
extern "C" fn check_running_state(
    _: *mut whisper_rs_sys::whisper_context,
    _: *mut whisper_rs_sys::whisper_state,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let run_transcription = AtomicBool::from_ptr(user_data as *mut bool);
        run_transcription.load(Ordering::Acquire)
    }
}
