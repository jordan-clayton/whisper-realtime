use std::fs::File;
use std::path::Path;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::formats::{FormatReader, Track};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

use crate::transcriber::static_transcriber::SupportedAudioSample;
use crate::utils::constants;
use crate::utils::errors::WhisperRealtimeError;

// TODO: test this implementation

// This is to restrict the audio to supported formats.
pub enum AudioSample<'a> {
    I16(&'a [i16]),
    F32(&'a [f32]),
    F64(&'a [f64]),
}

/// Resamples audio to 16kHz for Whisper processing
/// Packages into a SupportedAudioSample which can be passed to a StaticTranscriber
/// Audio will be converted to f32, because it is the most convenient
pub fn resample(
    samples: &AudioSample,
    out_sample_rate: f64,
    in_sample_rate: f64,
    num_channels: usize,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    if num_channels == 0 {
        return Err(WhisperRealtimeError::ParameterError(
            "Zero channels.".to_owned(),
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

    let samples_to_process = match samples {
        AudioSample::I16(audio_in) => {
            let mut output = vec![0.0f32; audio_in.len()];
            whisper_rs::convert_integer_to_float_audio(*audio_in, &mut output)?;
            output
        }
        AudioSample::F32(audio_in) => audio_in.to_vec(),
        AudioSample::F64(audio_in) => audio_in.iter().map(|s| *s as f32).collect(),
    };

    let resampler = SincFixedIn::new(
        out_sample_rate / in_sample_rate,
        2.0,
        params,
        samples_to_process.len() / num_channels,
        num_channels,
    )?;

    if num_channels == 2 {
        resample_stereo(samples_to_process, resampler)
    } else {
        resample_mono(samples_to_process, resampler)
    }
}

// These two functions have to take ownership of the resources for resampling
// The original slice provided is untouched; these work on a copy of the samples.
#[inline]
pub fn resample_mono(
    samples: Vec<f32>,
    mut resampler: SincFixedIn<f32>,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let waves_in = vec![samples];
    let waves_out = resampler.process(&waves_in, None)?;
    Ok(SupportedAudioSample::F32(waves_out[0].to_vec()))
}

#[inline]
pub fn resample_stereo(
    samples: Vec<f32>,
    mut resampler: SincFixedIn<f32>,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let waves_in: Vec<Vec<f32>> = vec![
        samples.iter().step_by(2).copied().collect(),
        samples[1..].iter().step_by(2).copied().collect(),
    ];

    let waves_out = resampler.process(&waves_in, None)?;

    let left = &waves_out[0];
    let right = &waves_out[1];
    let interleaved: Vec<f32> = left
        .iter()
        .zip(right)
        .map(|(x, y)| [*x, *y])
        .flatten()
        .collect();

    Ok(SupportedAudioSample::F32(interleaved))
}

// Since this is for whisper, it will also convert the samples to mono
#[inline]
pub fn normalize_audio(
    samples: &AudioSample,
    in_sample_rate: f64,
    num_channels: usize,
) -> Result<SupportedAudioSample, WhisperRealtimeError> {
    let resampled = resample(samples, 16000., in_sample_rate, num_channels)?;
    if let SupportedAudioSample::F32(stereo) = resampled {
        let mono = whisper_rs::convert_stereo_to_mono_audio(&stereo)?;
        Ok(SupportedAudioSample::F32(mono))
    } else {
        // This should never, ever happen
        Err(WhisperRealtimeError::ParameterError(
            "Resampling returned invalid audio format".to_owned(),
        ))
    }
}

// Probe and return a boolean
pub fn file_needs_normalizing<P: AsRef<Path>>(path: P) -> Result<bool, WhisperRealtimeError> {
    let file = Box::new(File::open(path)?);
    let mss = MediaSourceStream::new(file, Default::default());
    let hint = Hint::new();
    let format_opts = Default::default();
    let metadata_opts = Default::default();
    let probed =
        symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

    let format = probed.format;
    let track = format
        .default_track()
        .ok_or(WhisperRealtimeError::ParameterError(
            "Failed to get default track".to_owned(),
        ))?;
    needs_normalizing(track)
}

// Possibly just inline the codec_params instead of accessing.
#[inline]
pub fn needs_normalizing(track: &Track) -> Result<bool, WhisperRealtimeError> {
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or(WhisperRealtimeError::ParameterError(
            "Failed to get sample rate".to_owned(),
        ))? as f64;
    Ok(sample_rate != constants::WHISPER_SAMPLE_RATE)
}
