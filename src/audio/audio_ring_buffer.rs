use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use sdl2::audio::AudioFormatNum;

use crate::utils::constants;
use crate::utils::errors::WhisperRealtimeError;

/// This is a workaround for trait aliasing until nightly moves to stable.
pub trait AudioSampleFormat: Default + Clone + Copy + AudioFormatNum + 'static {}
impl<T: Default + Clone + Copy + AudioFormatNum + 'static> AudioSampleFormat for T {}

struct InnerAudioRingBuffer<T: AudioSampleFormat> {
    head: AtomicUsize,
    audio_len: AtomicUsize,
    capacity_ms: AtomicUsize,
    buffer_len: AtomicUsize,
    sample_rate: AtomicUsize,
    buffer: Mutex<Vec<T>>,
}

// TODO: documentation
#[derive(Clone)]
pub struct AudioRingBuffer<T: AudioSampleFormat> {
    inner: Arc<InnerAudioRingBuffer<T>>,
}

#[derive(Clone)]
pub struct AudioRingBufferBuilder<T: AudioSampleFormat> {
    // Size of the buffer in milliseconds
    capacity_ms: Option<usize>,
    sample_rate: Option<usize>,
    _marker: PhantomData<T>,
}

impl<T: AudioSampleFormat> AudioRingBufferBuilder<T> {
    pub fn new() -> Self {
        Self {
            capacity_ms: None,
            sample_rate: None,
            _marker: PhantomData,
        }
    }
    pub fn with_capacity_ms(mut self, capacity_ms: usize) -> Self {
        self.capacity_ms = Some(capacity_ms);
        self
    }
    pub fn with_sample_rate(mut self, sample_rate: usize) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    // This will fail to build if the length/sample rate are missing or zero.
    pub fn build(self) -> Result<AudioRingBuffer<T>, WhisperRealtimeError> {
        let c_ms =
            self.capacity_ms
                .filter(|&ms| ms > 0)
                .ok_or(WhisperRealtimeError::ParameterError(
                    "AudioRingBufferBuilder has zero-length buffer.".to_string(),
                ))?;
        let s_rate =
            self.sample_rate
                .filter(|&sr| sr > 0)
                .ok_or(WhisperRealtimeError::ParameterError(
                    " AudioRingBufferBuilder has zero-size sample rate.".to_string(),
                ))?;

        let capacity_ms = AtomicUsize::new(c_ms);
        let buffer_size = ((c_ms as f64 / 1000.) * (s_rate as f64)) as usize;
        let buffer_len = AtomicUsize::new(buffer_size);
        let audio_len = AtomicUsize::new(0);
        let head = AtomicUsize::new(0);
        let sample_rate = AtomicUsize::new(s_rate);
        let buffer = Mutex::new(vec![T::default(); buffer_size]);
        let inner = Arc::new(InnerAudioRingBuffer {
            head,
            audio_len,
            capacity_ms,
            buffer_len,
            sample_rate,
            buffer,
        });

        Ok(AudioRingBuffer { inner })
    }
}

impl<T: AudioSampleFormat> AudioRingBuffer<T> {
    /// The audio length is measured in sizeof(T)
    pub fn get_audio_length(&self) -> usize {
        self.inner.audio_len.load(Ordering::Acquire)
    }

    pub fn get_audio_length_ms(&self) -> usize {
        let audio_len = self.inner.audio_len.load(Ordering::Acquire) as f64;
        let sample_rate = self.inner.sample_rate.load(Ordering::Acquire) as f64;
        ((audio_len * 1000f64) / sample_rate) as usize
    }
    pub fn get_capacity_in_ms(&self) -> usize {
        self.inner.capacity_ms.load(Ordering::Acquire)
    }
    pub fn get_buffer_length(&self) -> usize {
        self.inner.buffer_len.load(Ordering::Acquire)
    }
    pub fn get_head_position(&self) -> usize {
        self.inner.head.load(Ordering::Acquire)
    }

    pub fn push_audio(&self, input: &[T]) {
        let len = input.len();
        let mut n_samples = len;
        let mut stream = input.to_vec();

        let buffer_len = self.inner.buffer_len.load(Ordering::Acquire);
        if n_samples > buffer_len {
            n_samples = buffer_len;
            let new_start = len - n_samples;
            stream = stream[new_start..].to_vec();
        }

        // Grab the buffer to hold the state before grabbing the head position
        let mut buffer = self.inner.buffer.lock();
        let head_pos = self.inner.head.load(Ordering::Acquire);
        if head_pos + n_samples > buffer_len {
            let offset = buffer_len - head_pos;
            // memcpy stuff
            // First copy.
            let copy_buffer = &mut buffer[head_pos..head_pos + offset];
            let stream_buffer = &stream[0..offset];

            copy_buffer.copy_from_slice(stream_buffer);

            // Second copy
            let diff_offset = n_samples - offset;
            let copy_buffer = &mut buffer[0..diff_offset];
            let stream_buffer = &stream[offset..offset + diff_offset];

            copy_buffer.copy_from_slice(stream_buffer);

            let new_head_pos = (head_pos + n_samples) % buffer_len;
            self.inner.head.store(new_head_pos, Ordering::Release);
            self.inner.audio_len.store(buffer_len, Ordering::Release);
        } else {
            let copy_buffer = &mut buffer[head_pos..head_pos + n_samples];
            let stream_buffer = &stream[0..n_samples];

            copy_buffer.copy_from_slice(stream_buffer);

            let new_head_pos = (head_pos + n_samples) % buffer_len;
            self.inner.head.store(new_head_pos, Ordering::Release);

            let audio_len = self.inner.audio_len.load(Ordering::Acquire);

            let new_audio_len = std::cmp::min(audio_len + n_samples, buffer_len);
            self.inner.audio_len.store(new_audio_len, Ordering::Release);
        }
    }

    pub fn read(&self, len_ms: usize) -> Vec<T> {
        let mut buf = vec![];
        self.read_into(len_ms, &mut buf);
        buf
    }

    pub fn read_into(&self, len_ms: usize, result: &mut Vec<T>) {
        let mut ms = len_ms;

        if ms == 0 {
            ms = self.inner.capacity_ms.load(Ordering::Acquire);
        }

        result.clear();
        let sample_rate = self.inner.sample_rate.load(Ordering::Acquire);
        let mut n_samples = (ms as f64 * sample_rate as f64 / 1000f64) as usize;

        // Grab the buffer to hold the state before checking the audio length.
        let buffer = self.inner.buffer.lock();

        let audio_len = self.inner.audio_len.load(Ordering::Acquire);
        if n_samples > audio_len {
            n_samples = audio_len;
        }
        result.resize(n_samples, T::default());
        // If n_samples == 0 (ie. the audio buffer has just been cleared).
        if result.len() == 0 {
            return;
        }

        let head_pos = self.inner.head.load(Ordering::Acquire);
        let buffer_len = self.inner.buffer_len.load(Ordering::Acquire);

        let mut start_pos: i64 = head_pos as i64 - n_samples as i64;

        if start_pos < 0 {
            start_pos += buffer_len as i64;
        }

        let start_pos = start_pos as usize;

        if start_pos + n_samples > buffer_len {
            let to_endpoint = buffer_len - start_pos;

            // First copy: starting pos (head - n_samples) up to the end of the buffer
            let copy_buffer = &mut result[0..to_endpoint];
            let stream = &buffer[start_pos..start_pos + to_endpoint];

            copy_buffer.copy_from_slice(stream);

            // second copy: start of the buffer up to the end of the remaining samples.
            let remaining_samples = n_samples - to_endpoint;
            let copy_buffer = &mut result[to_endpoint..to_endpoint + remaining_samples];
            let stream = &buffer[0..remaining_samples];

            copy_buffer.copy_from_slice(stream);
        } else {
            let copy_buffer = &mut result[0..n_samples];
            let stream = &buffer[start_pos..start_pos + n_samples];

            copy_buffer.copy_from_slice(stream);
        }
    }
    // Move the head and update audio length to 0
    // New data will overwrite old data, and grabbing audio from an empty ringbuffer will just
    // result in an empty vector
    pub fn clear(&self) {
        // Guard the state by hogging the mutex until atomic operations are done
        let _buffer = self.inner.buffer.lock();
        self.inner.head.store(0, Ordering::SeqCst);
        self.inner.audio_len.store(0, Ordering::SeqCst);
    }

    // Clear the requested amount of audio data from the back of the buffer,
    // minus a small amount of audio to try and resolve word boundaries.
    // Data is not zeroed out, but it will be overwritten when new samples are added to the buffer.
    pub fn clear_n_samples(&self, len_ms: usize) {
        // Guard the state by hogging the mutex until atomic operations are done
        let _buffer = self.inner.buffer.lock();
        let mut ms = len_ms.clone();
        if ms == 0 {
            let this_ms = self.inner.capacity_ms.load(Ordering::Acquire);
            ms = this_ms;
        }

        let sample_rate = self.inner.sample_rate.load(Ordering::Acquire);
        let mut n_samples = (ms as f64 * sample_rate as f64 / 1000f64) as usize;

        let audio_len = self.inner.audio_len.load(Ordering::Acquire);
        if n_samples > audio_len {
            n_samples = audio_len;
        }

        let new_len = audio_len - n_samples + constants::N_SAMPLES_KEEP;
        self.inner.audio_len.store(new_len, Ordering::Release);
    }
}

impl<T: AudioSampleFormat> Default for AudioRingBuffer<T> {
    fn default() -> Self {
        AudioRingBufferBuilder::new()
            .with_capacity_ms(constants::INPUT_BUFFER_CAPACITY)
            .with_sample_rate(constants::WHISPER_SAMPLE_RATE as usize)
            .build()
            .expect("Default AudioRingbuffer should build without problems.")
    }
}
