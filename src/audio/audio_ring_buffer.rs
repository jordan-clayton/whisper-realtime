use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use sdl2::audio::AudioFormatNum;

use crate::utils::constants;

pub struct AudioRingBuffer<T: Default + Clone + Copy + AudioFormatNum + 'static> {
    head: AtomicUsize,
    audio_len: AtomicUsize,
    capacity_ms: AtomicUsize,
    buffer_len: AtomicUsize,
    buffer: Mutex<Vec<T>>,
}

// The ring buffer is internally protected with thread-safe guards, and is therefore thread-safe
unsafe impl<T: Default + Clone + Copy + AudioFormatNum + 'static> Sync for AudioRingBuffer<T> {}

unsafe impl<T: Default + Clone + Copy + AudioFormatNum + 'static> Send for AudioRingBuffer<T> {}

/// A zero-length / zero-sample-rate buffer is considered invalid
/// Sending invalid parameters will return None
impl<T: Default + Clone + Copy + AudioFormatNum + 'static> AudioRingBuffer<T> {
    // A 1 second @ WHISPER_SAMPLE_RATE buffer
    pub fn new() -> Self {
        let buffer_size = constants::WHISPER_SAMPLE_RATE as usize;
        let head = AtomicUsize::new(0);
        let audio_len = AtomicUsize::new(0);
        let capacity_ms = AtomicUsize::new(1000);
        let buffer_len = AtomicUsize::new(buffer_size);
        Self {
            head,
            audio_len,
            capacity_ms,
            buffer_len,
            buffer: Mutex::new(vec![T::default(); buffer_size]),
        }
    }
    // Buffer size is in frames. A frame is 1 second @ sample rate in size.
    pub fn new_with_size(capacity_ms: usize, sample_rate: Option<f64>) -> Option<Self> {
        if capacity_ms < 1 {
            return None;
        }
        let s_rate = sample_rate.unwrap_or(constants::WHISPER_SAMPLE_RATE);
        let buffer_size = capacity_ms as f64 / 1000. * s_rate;
        let buffer_size = buffer_size as usize;
        let buffer_len = AtomicUsize::new(buffer_size);
        let head = AtomicUsize::new(0);
        let audio_len = AtomicUsize::new(0);
        let capacity_ms = AtomicUsize::new(capacity_ms);

        Some(Self {
            head,
            audio_len,
            capacity_ms,
            buffer_len,
            buffer: Mutex::new(vec![T::default(); buffer_size]),
        })
    }

    pub fn with_capacity_ms(self, capacity_ms: usize) -> Option<Self> {
        if capacity_ms < 1 {
            return None;
        }
        let buffer_size = self.buffer_len.load(Ordering::Acquire) as f64;
        let l = self.capacity_ms.load(Ordering::Acquire) as f64;
        // This will panic on zero-division
        let sample_rate = Some((buffer_size * 1000.) / l);
        Self::new_with_size(capacity_ms, sample_rate)
    }

    pub fn with_sample_rate(self, sample_rate: f64) -> Option<Self> {
        if sample_rate <= 0.0 {
            return None;
        }
        let len_ms = self.capacity_ms.load(Ordering::Acquire);
        Self::new_with_size(len_ms, Some(sample_rate))
    }
    pub fn audio_len(&self) -> usize {
        self.audio_len.load(Ordering::Acquire)
    }
    pub fn len_ms(&self) -> usize {
        self.capacity_ms.load(Ordering::Acquire)
    }
    pub fn buffer_len(&self) -> usize {
        self.buffer_len.load(Ordering::Acquire)
    }

    pub fn get_head_position(&self) -> usize {
        self.head.load(Ordering::Acquire)
    }

    pub fn push_audio(&self, input: &[T]) {
        let len = input.len();
        let mut n_samples = len.clone();
        let mut stream = input.to_vec();

        let buffer_len = self.buffer_len.load(Ordering::Acquire);
        if n_samples > buffer_len {
            n_samples = buffer_len;
            let new_start = len - n_samples;
            stream = stream[new_start..].to_vec();
        }

        // Grab the buffer to hold the state before grabbing the head position
        let buffer = self.buffer.lock();
        let mut buffer = if let Err(e) = buffer {
            eprintln!("Buffer mutex poisoned: {}", e);
            self.buffer.clear_poison();
            e.into_inner()
        } else {
            buffer.unwrap()
        };
        let head_pos = self.head.load(Ordering::Acquire);
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
            self.head.store(new_head_pos, Ordering::Release);
            self.audio_len.store(buffer_len, Ordering::Release);
        } else {
            let copy_buffer = &mut buffer[head_pos..head_pos + n_samples];
            let stream_buffer = &stream[0..n_samples];

            copy_buffer.copy_from_slice(stream_buffer);

            let new_head_pos = (head_pos + n_samples) % buffer_len;
            self.head.store(new_head_pos, Ordering::Release);

            let audio_len = self.audio_len.load(Ordering::Acquire);

            let new_audio_len = std::cmp::min(audio_len + n_samples, buffer_len);
            self.audio_len.store(new_audio_len, Ordering::Release);
        }
    }
    pub fn get_audio(&self, len_ms: usize, result: &mut Vec<T>) {
        let mut ms = len_ms.clone();

        if ms == 0 {
            let this_ms = self.capacity_ms.load(Ordering::Acquire);
            ms = this_ms;
        }

        result.clear();

        let mut n_samples = (ms as f64 * constants::WHISPER_SAMPLE_RATE / 1000f64) as usize;

        // Grab the buffer to hold the state before checking the audio length.
        let buffer = self.buffer.lock();
        let buffer = if let Err(e) = buffer {
            eprintln!("Buffer is poisoned: {}", e);
            self.buffer.clear_poison();
            e.into_inner()
        } else {
            buffer.unwrap()
        };

        let audio_len = self.audio_len.load(Ordering::Acquire);
        if n_samples > audio_len {
            n_samples = audio_len;
        }
        result.resize(n_samples, T::default());

        let head_pos = self.head.load(Ordering::Acquire);
        let buffer_len = self.buffer_len.load(Ordering::Acquire);

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
        let _buffer = self.buffer.lock();
        self.head.store(0, Ordering::SeqCst);
        self.audio_len.store(0, Ordering::SeqCst);
    }

    // Clear the requested amount of audio data, minus a small amount of audio to try and resolve
    // word boundaries.
    // Data is not zeroed out, but it will be overwritten when new samples are added to the buffer.
    pub fn clear_n_samples(&self, len_ms: usize) {
        // Guard the state by hogging the mutex until atomic operations are done
        let _buffer = self.buffer.lock();
        let mut ms = len_ms.clone();
        if ms == 0 {
            let this_ms = self.capacity_ms.load(Ordering::Acquire);
            ms = this_ms;
        }

        let mut n_samples = (ms as f64 * constants::WHISPER_SAMPLE_RATE / 1000f64) as usize;
        let audio_len = self.audio_len.load(Ordering::Acquire);
        if n_samples > audio_len {
            n_samples = audio_len;
        }

        let head_pos = self.head.load(Ordering::Acquire);
        let buffer_len = self.buffer_len.load(Ordering::Acquire);

        let mut start_pos: i64 =
            head_pos as i64 - n_samples as i64 + constants::N_SAMPLES_KEEP as i64;

        if start_pos < 0 {
            start_pos += buffer_len as i64;
        }

        let start_pos = start_pos as usize;

        // Move the head back to the start pos and update the length of the audio buffer
        let new_len = audio_len - n_samples + constants::N_SAMPLES_KEEP;
        self.audio_len.store(new_len, Ordering::Release);
        self.head.store(start_pos, Ordering::Release);
    }
}
