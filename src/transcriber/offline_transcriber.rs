use std::ffi::{c_int, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use whisper_rs::WhisperProgressCallback;

use crate::audio::{AudioChannelConfiguration, WhisperAudioSample};
use crate::transcriber::{
    CallbackTranscriber, OfflineWhisperProgressCallback, Transcriber, WhisperCallbacks,
    WhisperOutput,
};
use crate::transcriber::vad::VAD;
use crate::utils::errors::WhisperRealtimeError;
use crate::utils::Sender;
use crate::whisper::configs::WhisperConfigsV2;

/// Builder for [crate::transcriber::offline_transcriber::OfflineTranscriber]
/// Silero: [crate::transcriber::vad::Silero] is recommended for accuracy.
#[derive(Clone)]
pub struct OfflineTranscriberBuilder<V>
where
    V: VAD<f32>,
{
    configs: Option<Arc<WhisperConfigsV2>>,
    /// (Optional) Used for sending transcribed segments to a UI.
    sender: Option<Sender<WhisperOutput>>,
    audio: Option<Arc<WhisperAudioSample>>,
    channels: Option<AudioChannelConfiguration>,
    /// (Optional) Used to extract voiced segments to reduce overall transcription time.
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
    /// Sets the whisper configurations
    pub fn with_configs(mut self, configs: WhisperConfigsV2) -> Self {
        self.configs = Some(Arc::new(configs));
        self
    }
    /// Sets an (optional) channel through which to send transcribed segments to a UI
    pub fn with_sender(mut self, sender: Sender<WhisperOutput>) -> Self {
        self.sender = Some(sender);
        self
    }

    /// Sets the audio to be transcribed
    pub fn with_audio(mut self, audio: WhisperAudioSample) -> Self {
        self.audio = Some(Arc::new(audio));
        self
    }

    /// Sets the audio channel configurations.
    /// NOTE: This must match the audio source channel configurations or there will be transcription artefacts.
    pub fn with_channel_configurations(mut self, channels: AudioChannelConfiguration) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Sets an optional voice activity detector to optimize transcription by pruning out unvoiced audio frames
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
    /// Builds an OfflineTranscriber<V> according to the given parameters
    /// # Returns:
    /// * Ok(OfflineTranscriber<V>) on successful build
    /// * Err(WhisperRealtimeError) if one of the following are true:
    /// ** missing whisper configurations,
    /// ** missing channel configurations,
    /// ** missing audio
    pub fn build(self) -> Result<OfflineTranscriber<V>, WhisperRealtimeError> {
        let configs = self.configs.ok_or(WhisperRealtimeError::ParameterError(
            "Configs missing in OfflineTranscriberBuilder..".to_string(),
        ))?;
        let sender = self.sender;
        let audio = self.audio.filter(|audio| audio.len() > 0).ok_or(
            WhisperRealtimeError::ParameterError(
                "Audio missing in OfflineTranscriberBuilder.".to_string(),
            ),
        )?;
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

/// For running offline (non-realtime) transcription using whisper.
/// NOTE: timestamps have not yet been implemented.
#[derive(Clone)]
pub struct OfflineTranscriber<V: VAD<f32>> {
    /// Whisper configurations
    configs: Arc<WhisperConfigsV2>,
    /// (Optional) Used for sending transcribed segments to a UI.
    /// Segments are sent as [crate::transcriber::WhisperOutput::ConfirmedTranscription]
    sender: Option<Sender<WhisperOutput>>,
    /// The audio to transcribe.
    audio: Arc<WhisperAudioSample>,
    /// Mono or Stereo. Stereo will be converted to mono before transcription
    channels: AudioChannelConfiguration,
    /// (Optional) Used to extract voiced segments to reduce overall transcription time.
    voice_activity_detector: Option<Arc<Mutex<V>>>,
}

impl<V: VAD<f32>> OfflineTranscriber<V> {
    fn run_transcription(
        &mut self,
        full_params: whisper_rs::FullParams,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, WhisperRealtimeError> {
        let whisper_context_params = self.configs.to_whisper_context_params();
        let file_path_string = self.configs.model().file_path_string()?;
        // Set up a whisper context
        let ctx =
            whisper_rs::WhisperContext::new_with_params(&file_path_string, whisper_context_params)?;

        let mut whisper_state = ctx.create_state()?;

        // Prepare audio
        let mut audio_samples = match self.audio.as_ref() {
            WhisperAudioSample::I16(audio) => {
                let len = audio.len();
                let mut float_samples = vec![0.0; len];
                whisper_rs::convert_integer_to_float_audio(audio, &mut float_samples)?;
                float_samples.into_boxed_slice()
            }
            WhisperAudioSample::F32(audio) => audio.clone(),
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
            AudioChannelConfiguration::Mono => audio_samples.to_vec(),
            AudioChannelConfiguration::Stereo => {
                whisper_rs::convert_stereo_to_mono_audio(&audio_samples)?
            }
        };

        if let Err(e) = whisper_state.full(full_params, &mono_audio) {
            // Only escape early if the transcription is still supposed to be running;
            // Otherwise, the abort callback fired true, and run_transcription is false - indicating
            // the user has stopped the transcription.
            if run_transcription.load(Ordering::Acquire) {
                return Err(WhisperRealtimeError::WhisperError(e));
            }
        }

        // Otherwise, expect the transcription to have been successful; subsequent errors
        // will bubble up.
        let num_segments = whisper_state.full_n_segments()?;
        let mut text: Vec<String> = vec![];
        text.reserve(num_segments as usize);

        // Transcribe segments and send through a channel to a UI.
        for i in 0..num_segments {
            if let Ok(segment) = whisper_state.full_get_segment_text(i) {
                text.push(segment.clone());
                // This function doesn't need to stop if the channel is closed, so just ignore.
                let _ = self.sender.as_ref().and_then(|sender| {
                    // Since all segments are guaranteed not to overlap (eg. as in realtime), these
                    // can be sent as single segment strings to the receiver to handle.

                    // NOTE: this is optional because by this time, the full inference has been run
                    // Sending data out is a little redundant, but can be used in the UI to alert
                    // the user that the transcription is nearly finished.
                    Some(sender.send(WhisperOutput::ConfirmedTranscription(segment)))
                });
            }
        }

        // Clean up the whisper context
        drop(whisper_state);
        drop(ctx);
        // Return the final transcription.
        Ok(text.join("").trim().to_string())
    }
}

impl<V: VAD<f32>> Transcriber for OfflineTranscriber<V> {
    fn process_audio(
        &mut self,
        run_transcription: Arc<AtomicBool>,
    ) -> Result<String, WhisperRealtimeError> {
        let confs = Arc::clone(&self.configs);
        let mut full_params = confs.to_whisper_full_params();
        // Abort callback
        let r_transcription = Arc::clone(&run_transcription);

        // Coerce to a void pointer
        let a_ptr = Arc::into_raw(r_transcription) as *mut c_void;
        unsafe {
            full_params.set_abort_callback_user_data(a_ptr);
            full_params.set_abort_callback(Some(abort_callback))
        }

        let res = self.run_transcription(full_params, Arc::clone(&run_transcription));

        // Since the Arc is peeked in the C callback, a_ptr needs to be consumed one last time
        // to prevent memory leaks.
        unsafe {
            let _ = Arc::from_raw(a_ptr);
        }
        res
    }
}

impl<V, P> CallbackTranscriber<P> for OfflineTranscriber<V>
where
    V: VAD<f32>,
    P: OfflineWhisperProgressCallback,
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

        // Named stack binding for the progress callback
        let mut p_callback;

        // Abort callback
        let r_transcription = Arc::clone(&run_transcription);
        // Coerce to a void pointer
        let a_ptr = Arc::into_raw(r_transcription) as *mut c_void;

        let (progress_callback, user_data): (WhisperProgressCallback, *mut c_void) =
            match maybe_progress_callback {
                None => (None, std::ptr::null_mut::<c_void>()),
                Some(cb) => {
                    p_callback = cb;
                    (
                        Some(progress_callback::<P>),
                        &mut p_callback as *mut P as *mut c_void,
                    )
                }
            };
        unsafe {
            full_params.set_progress_callback_user_data(user_data);
            full_params.set_progress_callback(progress_callback);
            full_params.set_abort_callback_user_data(a_ptr);
            full_params.set_abort_callback(Some(abort_callback))
        }
        let res = self.run_transcription(full_params, Arc::clone(&run_transcription));
        // Since the Arc is peeked in the C callback, a_ptr needs to be consumed one last time
        // to prevent memory leaks.
        unsafe {
            let _ = Arc::from_raw(a_ptr);
        }

        res
    }
}

// C-Callbacks (until "safe" handles are working in whisper-rs)
// More callbacks will be implemented and exposed as necessary.

/// This function gets called at the beginning of each run of the decoder to determine whether to
/// abort transcription. The function aborts on true.
/// Since transcription is being controlled by a run_transcription boolean, the transcription
/// is expected to stop when run_transcription is false.
/// Internally, this peeks the Arc so that the refcount remains where it is.
/// Thus, the calling scope must manually consume the arc after this callback is no longer called.
unsafe extern "C" fn abort_callback(user_data: *mut c_void) -> bool {
    // Consume the pointer
    let ptr = unsafe { Arc::from_raw(user_data as *const AtomicBool) };
    // let arc = Arc::clone(&ptr);
    let run_transcription = ptr.load(Ordering::Acquire);
    // Prevent the refcount from decrementing
    let _ = Arc::into_raw(ptr);
    !run_transcription
}

/// This callback gets called in order to forward progress updates from Whisper to a UI.
/// To guarantee the safety of the C library, whisper_context and whisper_states should
/// not be mutated.
/// This function may or may not be called from a multithreaded context.
unsafe extern "C" fn progress_callback<PC: OfflineWhisperProgressCallback>(
    _: *mut whisper_rs_sys::whisper_context,
    _: *mut whisper_rs_sys::whisper_state,
    progress: c_int,
    user_data: *mut c_void,
) {
    let callback = unsafe { &mut *(user_data as *mut PC) };
    callback.call(progress);
}
