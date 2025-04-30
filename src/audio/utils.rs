use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::Decoder;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatReader;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::{Hint, ProbeResult};

#[cfg(feature = "resampler")]
use crate::audio::resampler::{AudioSample, needs_normalizing, normalize_audio};
use crate::transcriber::static_transcriber::SupportedAudioSample;
use crate::utils::errors::WhisperRealtimeError;

// TODO: cleanup

fn get_audio_probe<P: AsRef<Path> + Sized>(path: P) -> Result<ProbeResult, WhisperRealtimeError> {
    let file = Box::new(File::open(path)?);
    let mss = MediaSourceStream::new(file, Default::default());
    let hint = Hint::new();
    let format_opts = Default::default();
    let metadata_opts = Default::default();
    let probe = symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    Ok(probe)
}

// TODO: maybe don't wrap in the struct at the end?
pub fn load_audio_file<P: AsRef<Path>>(
    path: P,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let decoder_opts = Default::default();
    let probed = get_audio_probe(path)?;

    let format = probed.format;
    let track = format
        .default_track()
        .ok_or(WhisperRealtimeError::ParameterError(
            "Failed to get default audio track".to_owned(),
        ))?;

    let decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;
    // Decode loop
    let samples = decode_loop(track.id, decoder, format);
    Ok(SupportedAudioSample::F32(samples))
}

// This will return f32 audio normalized for whisper
// TODO: maybe don't wrap in the struct at the end?
#[cfg(feature = "resampler")]
pub fn load_normalized_audio_file<P: AsRef<Path> + Sized>(
    path: P,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let decoder_opts = Default::default();
    let probed = get_audio_probe(path)?;
    let format = probed.format;
    let track = format
        .default_track()
        .ok_or(WhisperRealtimeError::ParameterError(
            "Failed to get default track".to_owned(),
        ))?
        .clone();

    // Get the codec parameters before passing ownership to the decode loop.
    let codec_params = &track.codec_params;
    let sample_rate = codec_params
        .sample_rate
        .ok_or(WhisperRealtimeError::ParameterError(
            "Failed to grab sample rate".to_owned(),
        ))? as f64;

    let num_channels = codec_params
        .channels
        .ok_or(WhisperRealtimeError::ParameterError(
            "Failed to grab number of channels".to_owned(),
        ))?
        .count();

    let needs_normalizing = needs_normalizing(&track);
    let decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;
    // Decode loop
    let samples = decode_loop(track.id, decoder, format);

    // Normalize
    if needs_normalizing? {
        let audio = AudioSample::F32(&samples);
        normalize_audio(&audio, sample_rate, num_channels)
    } else {
        Ok(SupportedAudioSample::F32(samples))
    }
}

// TODO: add decode_loop with progress callback
// TODO: clean this up
fn decode_loop(
    track_id: u32,
    mut decoder: Box<dyn Decoder>,
    mut reader: Box<dyn FormatReader>,
) -> Vec<f32> {
    let mut samples = vec![];
    let mut sample_buf = None;

    // TODO: error handling, remove the panic
    loop {
        let packet = match reader.next_packet() {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        };

        // Consume metadata; not really sure what to do with this.
        while !reader.metadata().is_latest() {
            reader.metadata().pop();
        }

        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buffer) => {
                if sample_buf.is_none() {
                    let spec = *audio_buffer.spec();
                    let duration = audio_buffer.capacity() as u64;
                    sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                }

                let in_mono = audio_buffer.spec().channels.iter().count() == 1;

                if let Some(buf) = &mut sample_buf {
                    if in_mono {
                        buf.copy_planar_ref(audio_buffer);
                    } else {
                        buf.copy_interleaved_ref(audio_buffer);
                    }

                    samples.extend_from_slice(buf.samples());
                }
            }
            // TODO: proper error handling
            Err(Error::DecodeError(_)) => (),
            Err(_) => break,
        };
    }
    samples
}
