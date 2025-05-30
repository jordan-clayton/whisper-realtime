pub mod audio_ring_buffer;
pub mod loading;
pub mod microphone;
pub mod pcm;
pub mod recorder;
#[cfg(feature = "resampler")]
pub mod resampler;

/// Encapsulates a slice of (supported-format) audio for whisper transcription.
pub enum WhisperAudioSample {
    I16(Box<[i16]>),
    F32(Box<[f32]>),
}
impl WhisperAudioSample {
    pub fn len(&self) -> usize {
        match self {
            WhisperAudioSample::I16(audio) => audio.len(),
            WhisperAudioSample::F32(audio) => audio.len(),
        }
    }
}

/// Encapsulates supported channel configurations
#[derive(Copy, Clone, PartialEq)]
pub enum AudioChannelConfiguration {
    Mono,
    Stereo,
}
