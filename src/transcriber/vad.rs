use std::f64::consts::PI;
use std::sync::Mutex;

use lazy_static::lazy_static;
use realfft::RealFftPlanner;
use voice_activity_detector::VoiceActivityDetector;

use crate::constants;
use crate::errors::{WhisperRealtimeError, WhisperRealtimeErrorType};

// This is for the naive strategy to avoid extra memory allocations at runtime.
lazy_static! {
    static ref FFT_Planner: Mutex<RealFftPlanner<f64>> = Mutex::new(RealFftPlanner::<f64>::new());
}

pub enum VadStrategy {
    Naive,
    Silero,
}

impl Default for VadStrategy {
    fn default() -> Self {
        Self::Naive
    }
}

pub trait VoiceActivityDetection<
    T: voice_activity_detector::Sample
        + num_traits::cast::NumCast
        + num_traits::cast::FromPrimitive
        + num_traits::cast::ToPrimitive
        + num_traits::Zero,
>
{
    fn is_voice_detected_silero(
        vad: &mut VoiceActivityDetector,
        samples: &Vec<T>,
        voice_probability_threshold: f32,
    ) -> bool {
        let samples = samples.clone();
        let probability = vad.predict(samples);
        probability > voice_probability_threshold
    }

    fn is_voice_detected_naive(
        sample_rate: f64,
        samples: &Vec<T>,
        voice_energy_threshold: f64,
        window_len: f64,
        window_step: f64,
        freq_threshold: f64,
        voice_probability_threshold: f32,
    ) -> Result<bool, WhisperRealtimeError> {
        let mut samples_f64: Vec<f64> = samples
            .iter()
            .map(|n| n.to_f64().expect("Failed to convert T to f64"))
            .collect();

        if freq_threshold > 0.0f64 {
            let original_rms = calculate_rms(&samples_f64);
            high_pass_filter(&mut samples_f64, freq_threshold, sample_rate);
            let filtered_rms = calculate_rms(&samples_f64);
            let gain = original_rms / filtered_rms;
            apply_gain(&mut samples_f64, gain);
        }

        // run a DFT and use short time energy based VAD.
        let dft_vad = naive_frame_energy_vad(
            &samples_f64,
            sample_rate,
            voice_energy_threshold,
            window_len,
            window_step,
            constants::E0,
        );

        if let Err(e) = dft_vad {
            return Err(e);
        }

        let vad = dft_vad.unwrap().0;

        let mean: f32 = vad.iter().fold(0.0, |acc, n| acc + *n as f32) / vad.len() as f32;
        Ok(mean > voice_probability_threshold)
    }
}

// This mutates the samples to remove low frequency sounds.
fn high_pass_filter(samples: &mut Vec<f64>, frequency_threshold: f64, sample_rate: f64) {
    let rc: f64 = 1.0 / (2.0 * PI * frequency_threshold);
    let dt: f64 = 1.0 / sample_rate;

    let alpha: f64 = dt / (rc + dt);

    let mut y = samples[0];

    for i in 1..samples.len() {
        y = alpha * (y + samples[i] - samples[i - 1]);

        samples[i] = y;
    }
}

// Audio volume correction after high-pass
fn calculate_rms(data: &[f64]) -> f64 {
    let sum_of_squares: f64 = data.iter().fold(0.0, |acc, x| acc + (*x).powi(2));
    (sum_of_squares / data.len() as f64).sqrt()
}

fn apply_gain(data: &mut [f64], gain: f64) {
    for x in data.iter_mut() {
        *x *= gain;
    }
}

// The naive VAD is a port of the code used in this blog post by Ayoub Malek:
// https://superkogito.github.io/blog/2020/02/09/naive_vad.html
//
// I do not have any knowledge in this domain yet, nor do I have much knowledge of
// python/numpy.

// At this time, generics are not needed & would complicate the implementation.

/// A basic  Median filter
fn medfilt(data: &[f64], kernel_size: usize) -> Vec<f64> {
    let mut filtered = vec![0.0; data.len()];
    let k = kernel_size / 2;
    for i in 0..data.len() {
        let start = if i < k { 0 } else { i - k };
        let end = if i + k >= data.len() {
            data.len() - 1
        } else {
            i + k
        };
        let mut window: Vec<f64> = data[start..=end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        filtered[i] = window[window.len() / 2];
    }
    filtered
}

fn repeat_elements(data: &[f64], repeat: usize) -> Vec<f64> {
    data.iter()
        .flat_map(|&x| std::iter::repeat(x).take(repeat))
        .collect()
}

//
fn stride_trick(array: &[f64], stride_length: usize, stride_step: usize) -> Vec<Vec<f64>> {
    let n = ((array.len() - stride_length) / stride_step) + 1;

    (0..n)
        .map(|i| array[i * stride_step..i * stride_step + stride_length].to_vec())
        .collect()
}

fn framing(
    samples: &[f64],
    sample_rate: f64,
    window_len: f64,
    window_step: f64,
) -> Result<(Vec<Vec<f64>>, usize), WhisperRealtimeError> {
    if window_len < window_step {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::ParameterError,
            format!(
                "Framing: window_len, {} must be larger than window_hop, {}",
                window_len, window_step
            ),
        ));
    }

    let frame_length = (window_len * sample_rate) as usize;
    let frame_step = (window_step * sample_rate) as usize;

    let signal_length = samples.len();
    let frames_overlap = frame_length - frame_step;

    let rest_samples =
        (signal_length.abs_diff(frames_overlap)) % (frame_length.abs_diff(frames_overlap));

    let pad_samples: Vec<f64> = samples
        .iter()
        .cloned()
        .chain(vec![0.0; frame_step - rest_samples].into_iter())
        .collect();

    let frames = stride_trick(&pad_samples, frame_length, frame_step);
    Ok((frames, frame_length))
}

fn calculate_normalized_short_time_energy(
    frames: &[Vec<f64>],
) -> Result<Vec<f64>, WhisperRealtimeError> {
    if frames.len() < 1 {
        return Err(WhisperRealtimeError::new(
            WhisperRealtimeErrorType::ParameterError,
            String::from("STE: cannot calculate from 0-length array"),
        ));
    }

    let planner = &mut FFT_Planner.lock().expect("Failed to get FFT mutex");

    let fft = planner.plan_fft_forward(frames[0].len());

    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    Ok(frames
        .iter()
        .map(|frame| {
            input.copy_from_slice(frame);
            fft.process(&mut input, &mut output)
                .expect("Failed to process fft");

            output.iter().map(|c| c.norm_sqr()).sum::<f64>() / (frame.len() as f64).powi(2)
        })
        .collect())
}

// Array of 0/1 and the vector of voice frames.
fn naive_frame_energy_vad(
    samples: &[f64],
    sample_rate: f64,
    threshold: f64,
    window_len: f64,
    window_hop: f64,
    e0: f64,
) -> Result<(Vec<u8>, Vec<f64>), WhisperRealtimeError> {
    let result = framing(samples, sample_rate, window_len, window_hop);

    if let Err(e) = result {
        return Err(e);
    }

    // So far, things work up to the frames.
    let (frames, frames_len) = result.unwrap();

    // Compute STE -> Voiced frames
    let normalized_energy = calculate_normalized_short_time_energy(&frames);
    if let Err(e) = normalized_energy {
        return Err(e);
    }
    let energy = normalized_energy.unwrap();

    // log_energy not calculated properly.
    let log_energy: Vec<f64> = energy.iter().map(|e| 10.0 * (*e / e0).log10()).collect();

    let filtered_energy = medfilt(&log_energy, constants::KERNEL_SIZE);
    let repeated_energy = repeat_elements(&filtered_energy, frames_len);

    let mut vad: Vec<u8> = vec![0; repeated_energy.len()];

    // f_frames should be equal in size to vad.
    let f_frames: Vec<f64> = frames.iter().flatten().map(|n| *n).collect();

    assert_eq!(
        vad.len(),
        f_frames.len(),
        "Frame padding is wrong. Vad Len: {} F_Frame Len: {}",
        &vad.len(),
        &f_frames.len()
    );

    let mut v_frames: Vec<f64> = Vec::new();
    for (i, &e) in repeated_energy.iter().enumerate() {
        if e > threshold {
            vad[i] = 1;
            v_frames.push(f_frames[i]);
        }
    }

    Ok((vad, v_frames))
}

// IMPLEMENTATION TESTS
// noinspection DuplicatedCode
#[cfg(test)]
mod vad_tests {
    use hound;
    use hound::SampleFormat;

    use crate::configs;

    use super::*;

    pub struct AudioTester;

    impl VoiceActivityDetection<f32> for AudioTester {}

    #[test]
    #[ignore]
    fn test_framing() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let configs = configs::Configs::default();
        let audio: Vec<f64> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f64
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized
                })
                .collect(),
        };

        let result = framing(
            &audio,
            sample_rate as f64,
            configs.naive_window_len,
            configs.naive_window_step,
        );
        assert!(result.is_ok(), "{}", result.err().unwrap().cause());

        let audio: Vec<f64> = result.unwrap().0.iter().flatten().map(|n| *n).collect();
        // Write the output in f32, regular spec.

        let write_spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut writer =
            hound::WavWriter::create("tests/audio_files/test_flatten_f32.wav", write_spec)
                .expect("Failed to create wav writer");

        for i in 0..audio.len() {
            writer
                .write_sample(audio[i] as f32)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize write");

        // De-normalize back to i16 & write
        let write_spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer =
            hound::WavWriter::create("tests/audio_files/test_flatten_i16.wav", write_spec)
                .expect("Failed to create wav writer");

        for i in 0..audio.len() {
            let sample = audio[i];
            let normalized = if sample.is_sign_positive() {
                (audio[i] * (i16::MAX as f64)) as i16
            } else {
                (-1f64 * audio[i] * (i16::MIN as f64)) as i16
            };

            writer
                .write_sample(normalized)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize write");
    }

    #[test]
    #[ignore]
    fn test_high_pass_output() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let configs = configs::Configs::default();
        let mut audio: Vec<f64> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f64
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized
                })
                .collect(),
        };

        let original_rms = calculate_rms(&audio);

        high_pass_filter(
            &mut audio,
            configs.naive_vad_freq_threshold,
            sample_rate as f64,
        );

        let filtered_rms = calculate_rms(&audio);
        let gain = original_rms / filtered_rms;

        apply_gain(&mut audio, gain);

        let write_spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut writer =
            hound::WavWriter::create("tests/audio_files/test_high_pass.wav", write_spec)
                .expect("Failed to create wav writer");

        for i in 0..audio.len() {
            writer
                .write_sample(audio[i] as f32)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize write");
    }

    #[test]
    #[ignore]
    fn test_naive_frame_energy_vad() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let audio: Vec<f64> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f64
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized
                })
                .collect(),
        };

        let mut configs = configs::Configs::default();
        // Given the length of the audio track, this is being reduced to avoid
        // unnecessary panicking.
        // The provided audio track has around 60% speech.
        configs.voice_probability_threshold = 0.5;

        let result = naive_frame_energy_vad(
            &audio,
            sample_rate as f64,
            configs.naive_vad_energy_threshold,
            configs.naive_window_len,
            configs.naive_window_step,
            constants::E0,
        );

        assert!(result.is_ok(), "{}", result.err().unwrap().cause());

        let (vad, v_frames) = result.unwrap();

        let mean: f32 = vad.iter().fold(0.0, |acc, n| acc + *n as f32) / vad.len() as f32;
        assert!(
            mean > configs.voice_probability_threshold,
            "Voice not properly detected. Computed mean: {} > Threshold: {}",
            mean,
            configs.voice_probability_threshold
        );

        assert!(v_frames.len() > 0, "No vframes detected");

        // Test writing - not working.
        let write_spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer =
            hound::WavWriter::create("tests/audio_files/test_naive_vframes_f32.wav", write_spec)
                .expect("Failed to create wav writer");

        for i in 0..v_frames.len() {
            writer
                .write_sample(v_frames[i] as f32)
                .expect("Failed to write sample");
        }
        writer.finalize().expect("Failed to finalize write");
    }
    #[test]
    #[ignore]
    fn test_naive_voice_detection() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let audio: Vec<f32> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f32
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized as f32
                })
                .collect(),
        };

        let mut configs = configs::Configs::default();
        // No high-pass
        configs.naive_vad_freq_threshold = 0.0;
        // The audio track contains around 60% speech.
        configs.voice_probability_threshold = 0.5;
        let result = AudioTester::is_voice_detected_naive(
            sample_rate as f64,
            &audio,
            configs.naive_vad_energy_threshold,
            configs.naive_window_len,
            configs.naive_window_step,
            configs.naive_vad_freq_threshold,
            configs.voice_probability_threshold,
        );

        assert!(result.is_ok(), "{}", result.err().unwrap().cause());

        let voice_detected = result.unwrap();
        assert!(voice_detected, "Failed to detect voice");
    }

    #[test]
    #[ignore]
    fn test_naive_voice_detection_with_high_pass() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let audio: Vec<f32> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f32
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized as f32
                })
                .collect(),
        };

        let mut configs = configs::Configs::default();

        // The audio track contains around 60% speech.
        configs.voice_probability_threshold = 0.5;
        let result = AudioTester::is_voice_detected_naive(
            sample_rate as f64,
            &audio,
            configs.naive_vad_energy_threshold,
            configs.naive_window_len,
            configs.naive_window_step,
            configs.naive_vad_freq_threshold,
            configs.voice_probability_threshold,
        );

        assert!(result.is_ok(), "{}", result.err().unwrap().cause());

        let voice_detected = result.unwrap();
        assert!(voice_detected, "Failed to detect voice");
    }

    // This is known to work, but for testing speed on cpu.
    #[test]
    #[ignore]
    fn test_silero_vad() {
        // 8000Hz, mono
        let mut reader =
            hound::WavReader::open("tests/audio_files/OSR_us_000_0060_8k.wav").unwrap();
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let sample_format = spec.sample_format;

        let audio: Vec<f32> = match sample_format {
            SampleFormat::Float => reader
                .samples::<f32>()
                .map(|s| {
                    let sample = s.unwrap();
                    sample as f32
                })
                .collect(),
            SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| {
                    let sample = s.unwrap();

                    // Normalize for f32 wav spec
                    let normalized = if sample.is_positive() {
                        (sample as f64) / (i16::MAX as f64)
                    } else {
                        -1f64 * (sample as f64) / (i16::MIN as f64)
                    };

                    normalized as f32
                })
                .collect(),
        };

        // This might not be mutable.
        let mut configs = configs::Configs::default();

        let mut vad = voice_activity_detector::VoiceActivityDetector::builder()
            .sample_rate(sample_rate as i64)
            .chunk_size(1024usize)
            .build()
            .expect("Failed to build voice activity detector");

        // Break into 10s chunks.
        let secs = 10;
        let audio_chunks = audio.chunks_exact(sample_rate as usize * secs);
        let len = audio_chunks.len();
        let sum: u64 = audio_chunks.fold(0, |acc, v| {
            if AudioTester::is_voice_detected_silero(
                &mut vad,
                &v.to_vec(),
                configs.voice_probability_threshold,
            ) {
                acc + 1
            } else {
                acc
            }
        });

        // Reducing the threshold a little bit
        // The audio track contains around 60% speech.
        configs.voice_probability_threshold = 0.5;

        let mean = sum as f64 / len as f64;
        assert!(
            mean > configs.voice_probability_threshold as f64,
            "Failed to detect voice. Mean: {}",
            mean
        );
    }

    #[test]
    #[ignore]
    fn speed_test() {
        let mut time = std::time::Instant::now();
        for _i in 0..1000 {
            test_naive_voice_detection();
        }
        let mut now = std::time::Instant::now();
        let naive_diff = (now - time).as_millis();

        let naive_avg = naive_diff as f64 / 1000f64;

        time = std::time::Instant::now();

        for _i in 0..1000 {
            test_naive_voice_detection_with_high_pass();
        }

        now = std::time::Instant::now();

        let high_pass_diff = (now - time).as_millis();

        let high_pass_avg = high_pass_diff as f64 / 1000f64;

        time = std::time::Instant::now();

        for _i in 0..1000 {
            test_silero_vad()
        }

        now = std::time::Instant::now();
        let silero_diff = (now - time).as_millis();

        let silero_avg = silero_diff as f64 / 1000f64;

        // Need to use --nocapture or --show-output to get this output.

        println!(
            "Per 1000 iterations: \n\
        Naive: {}ms\n\
        High-Pass: {}ms\n\
        Silero: {}ms",
            naive_avg, high_pass_avg, silero_avg
        );
    }
}
