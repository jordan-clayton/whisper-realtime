// TODO: this is hard to read; separate imports.
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    sync::Mutex,
};

use sdl2::audio::AudioFormatNum;

use crate::constants;

pub struct AudioRingBuffer<T: Default + Clone + Copy + AudioFormatNum + 'static> {
    head: AtomicUsize,
    // TODO: get rid of pub -> implement builder.
    pub audio_len: AtomicUsize,
    pub len_ms: AtomicUsize,
    pub buffer_len: AtomicUsize,
    buffer: Mutex<Vec<T>>,
}

// The ring buffer should be thread safe.
unsafe impl<T: Default + Clone + Copy + AudioFormatNum + 'static> Sync for AudioRingBuffer<T> {}
unsafe impl<T: Default + Clone + Copy + AudioFormatNum + 'static> Send for AudioRingBuffer<T> {}

impl<T: Default + Clone + Copy + AudioFormatNum + 'static> AudioRingBuffer<T> {
    // TODO: generalize this -> sample rate should be a parameter, abstract out constants::WHISPER_SAMPLE_RATE.
    // TODO: builder pattern; construct with default params and use methods to set attrs.
    pub fn new(len_ms: usize) -> Self {
        // Buffer size is in seconds; I am not entirely sure why. TODO: investigate this.
        let buffer_size = (len_ms / 1000) as f64 * constants::WHISPER_SAMPLE_RATE;
        let buffer_size = buffer_size as usize;
        let buffer_len = AtomicUsize::new(buffer_size);
        let head = AtomicUsize::new(0);
        let audio_len = AtomicUsize::new(0);
        let len_ms = AtomicUsize::new(len_ms);
        Self {
            head,
            audio_len,
            len_ms,
            buffer_len,
            buffer: Mutex::new(vec![T::default(); buffer_size]),
        }
    }

    pub fn get_head_position(&self) -> usize {
        self.head.load(Ordering::Acquire)
    }

    // TODO: I don't think this needs to be a mut slice.
    pub fn push_audio(&self, input: &mut [T]) {
        let len = input.len();
        let mut n_samples = len.clone();
        let mut stream = input.to_vec();

        let buffer_len = self.buffer_len.load(Ordering::Acquire);
        if n_samples > buffer_len {
            n_samples = buffer_len;
            let new_start = len - n_samples;
            stream = stream[new_start..].to_vec();
        }

        let mut buffer = self.buffer.lock().expect("Mutex poisoned");
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
            // This might have an off-by-one error
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
    // TODO: rename ms -> len_ms
    pub fn get_audio(&self, ms: usize, result: &mut Vec<T>) {
        let mut ms = ms.clone();
        if ms == 0 {
            let this_ms = self.len_ms.load(Ordering::Acquire);
            ms = this_ms;
        }

        result.clear();

        let mut n_samples = (ms as f64 * constants::WHISPER_SAMPLE_RATE / 1000f64) as usize;
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

        let buffer = self.buffer.lock().unwrap();
        if start_pos + n_samples > buffer_len {
            let to_endpoint = buffer_len - start_pos;

            // First copy
            let copy_buffer = &mut result[0..to_endpoint];
            let stream = &buffer[start_pos..start_pos + to_endpoint];

            copy_buffer.copy_from_slice(stream);

            // second copy
            let remaining_samples = n_samples - to_endpoint;
            let copy_buffer = &mut result[0..remaining_samples];
            let stream = &buffer[0..remaining_samples];

            copy_buffer.copy_from_slice(stream);
        } else {
            let copy_buffer = &mut result[0..n_samples];
            let stream = &buffer[start_pos..start_pos + n_samples];

            copy_buffer.copy_from_slice(stream);
        }
    }

    pub fn clear(&self) {
        let _buffer = self.buffer.lock().unwrap();
        self.head.store(0, Ordering::SeqCst);
        self.audio_len.store(0, Ordering::SeqCst);
    }

    pub fn clear_n_samples(&self, ms: usize) {
        let mut ms = ms.clone();
        if ms == 0 {
            let this_ms = self.len_ms.load(Ordering::Acquire);
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

        let mut buffer = self.buffer.lock().unwrap();
        if start_pos + n_samples > buffer_len {
            let to_endpoint = buffer_len - start_pos;

            // Zero-out first half
            let stream = &mut buffer[start_pos..start_pos + to_endpoint];
            stream.iter_mut().for_each(|m| *m = T::default());

            // Zero-out second half
            let remaining_samples = n_samples - to_endpoint;
            let stream = &mut buffer[0..remaining_samples];
            stream.iter_mut().for_each(|m| *m = T::default());
        } else {
            let stream = &mut buffer[start_pos..start_pos + n_samples];
            stream.iter_mut().for_each(|m| *m = T::default());
        }
    }
}
