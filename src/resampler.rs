use std::fs::File;
use std::path::Path;

use ort::MemoryType::Default;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::Decoder;
use symphonia::core::errors::Error;
use symphonia::core::formats::{FormatReader, Track};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::{Hint, ProbeResult};

use crate::constants;
// TODO: possibly make this into a feature instead of the core library.
use crate::errors::{WhisperRealtimeError, WhisperRealtimeErrorType};
use crate::transcriber::static_transcriber::SupportedAudioSample;

// TODO: possibly make into a struct with the audio information?
pub enum AudioSample<'a> {
    I16(&'a [i16]),
    F32(&'a [f32]),
    F64(&'a [f64]),
}

// Resamples audio to 16kHz for Whisper processing
// Packages into a SupportedAudioSample which can be passed to a StaticTranscriber
// Audio will be converted to f32, because it is the most convenient
pub fn resample(
    samples: &AudioSample,
    out_sample_rate: f64,
    in_sample_rate: f64,
    num_channels: usize,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    if num_channels == 0 {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::ParameterError,
            "Zero channels".to_owned(),
        ));
    }
    // TODO: check rubato examples
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // LOL, rename
    let samples_to_process = match samples {
        AudioSample::I16(audio_in) => {
            let mut output = vec![0.0f32; audio_in.len()];
            whisper_rs::convert_integer_to_float_audio(*audio_in, &mut output)
                .expect("Audio samples length doesn't match");
            output
        }
        AudioSample::F32(audio_in) => audio_in.to_vec(),
        AudioSample::F64(audio_in) => audio_in.iter().filter_map(|s| s.into()).collect(),
    };

    let mut resampler = SincFixedIn::new(
        out_sample_rate / in_sample_rate,
        2.0,
        params,
        samples_to_process.len() / num_channels,
        num_channels,
    );
    if let Err(_) = resampler.as_ref() {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::ParameterError,
            "Failed to create resample params".to_owned(),
        ));
    }

    if num_channels == 2 {
        resample_stereo(samples_to_process, resampler.unwrap())
    } else {
        resample_mono(samples_to_process, resampler.unwrap())
    }
}

// TODO: think about this, don't love it -> inline it if using.
// Not entirely sold. Have to take ownership
#[inline]
fn resample_mono(
    samples: Vec<f32>,
    mut resampler: SincFixedIn<f32>,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let waves_in = vec![samples];
    let waves_out = resampler.process(&waves_in, None);
    match waves_out.as_ref() {
        // ResampleError
        Err(_) => Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::Unknown,
            "Failed to resample".to_owned(),
        )),
        Ok(waves_out) => Ok(SupportedAudioSample::F32(waves_out[0].to_vec())),
    }
}

// TODO: inline or remove; not in love with this.
#[inline]
fn resample_stereo(
    samples: Vec<f32>,
    mut resampler: SincFixedIn<f32>,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let waves_in: Vec<Vec<f32>> = vec![
        samples.iter().step_by(2).collect(),
        samples[1].iter().step_by(2).collect,
    ];

    let waves_out = resampler.process(&waves_in, None);
    match waves_out.as_ref() {
        // ResampleError
        Err(_) => Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::Unknown,
            "Failed to resample".to_owned(),
        )),
        Ok(waves_out) => {
            let interleaved: Vec<f32> = &waves_out[0]
                .iter()
                .zip(&waves_out[1])
                .map(|(x, y)| [*x, *y])
                .flatten()
                .collect();

            Ok(SupportedAudioSample::F32(interleaved))
        }
    }
}

#[inline]
pub fn normalize_audio(
    samples: &AudioSample,
    in_sample_rate: f64,
    num_channels: usize,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    resample(samples, 16000., in_sample_rate, num_channels)
}

// TODO: move where appropriate, probably better to make a separate module
fn get_audio_probe<P: Into<Path>>(path: P) -> ProbeResult {
    let file = Box::new(File::open(path)?);
    let mss = MediaSourceStream::new(file, Default::default());
    let hint = Hint::new();
    let format_opts = Default::default();
    let metadata_opts = Default::default();
    symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?
}

pub fn load_audio_file<P: Into<Path>>(
    path: P,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let decoder_opts = Default::default();
    let probed = get_audio_probe(path);

    let format = probed.format;
    let track = format.default_track()?;

    let decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;
    // Decode loop
    let samples = decode_loop(track.id, decoder, format);
    Ok(SupportedAudioSample::F32(samples))
}

// This will return f32 audio normalized for whisper
// TODO: maybe don't wrap in the struct at the end?
pub fn load_normalized_audio_file<P: Into<Path>>(
    path: P,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let decoder_opts = Default::default();
    let probed = get_audio_probe(path);
    let format = probed.format;
    let track = format.default_track()?;
    let needs_normalizing = needs_normalizing(track);
    let decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;
    // Decode loop
    let samples = decode_loop(track.id, decoder, format);
    // Normalize
    if needs_normalizing {
        let audio = AudioSample::F32(&samples);
        normalize_audio(
            &audio,
            track.codec_params.sample_rate? as f64,
            track.codec_params.channels?.count(),
        )
    } else {
        Ok(SupportedAudioSample::F32(samples))
    }
}

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
            reader.metadata().pop()
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

                let in_mono = *audio_buffer.spec().channels.iter().count() == 1;

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

// Probe and return a boolean
pub fn file_needs_normalizing<P: Into<Path>>(path: P) -> bool {
    let file = Box::new(File::open(path)?);
    let mss = MediaSourceStream::new(file, Default::default());
    let hint = Hint::new();
    let format_opts = Default::default();
    let metadata_opts = Default::default();
    let probed =
        symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

    let mut format = probed.format;
    let track = format.default_track()?;
    needs_normalizing(track)
}

// Possibly just inline the codec_params instead of accessing.
#[inline]
pub fn needs_normalizing(track: &Track) -> bool {
    track.codec_params.sample_rate? as f64 != constants::WHISPER_SAMPLE_RATE
}
