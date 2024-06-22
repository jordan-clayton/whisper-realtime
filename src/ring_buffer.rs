use std::sync::Mutex;

use crate::constants;

pub struct AudioRingBuffer<T: Default + Clone + Copy> {
    head: usize,
    count: usize,
    buffer: Mutex<Vec<T>>,
}

// This is not ideal, but the ring-buffer should be thread-safe
unsafe impl<T: Default + Clone + Copy> Sync for AudioRingBuffer<T> {}
unsafe impl<T: Default + Clone + Copy> Send for AudioRingBuffer<T> {}

impl<T: Default + Clone + Copy> AudioRingBuffer<T> {
    pub fn new(size: usize) -> Self {
        AudioRingBuffer {
            head: 0,
            count: 0,
            buffer: Mutex::new(vec![T::default(); size]),
        }
    }

    pub fn push_audio(&mut self, data: &[T], len: usize) {
        let mut n_samples = len / std::mem::size_of::<T>();

        let mut stream = data.to_vec();

        if n_samples > self.count {
            n_samples = self.count;
            let new_len = stream.len() + (len - n_samples);
            stream.resize(new_len, T::default());
        }

        let buffer = self.buffer.get_mut().unwrap();

        if self.head + n_samples > buffer.len() {
            let wrap_around = self.count - self.head;

            // memcpy
            let (left_buffer, right_buffer) = buffer.split_at_mut(self.head);

            right_buffer.copy_from_slice(&stream[self.head..=wrap_around]);
            left_buffer.copy_from_slice(&stream[0..=n_samples - wrap_around]);

            self.head = (self.head + n_samples) % buffer.len();
            self.count = buffer.len();
        } else {
            // memcpy
            let mut copy_buffer = &mut buffer[self.head..];
            let copy_stream = &stream[0..=n_samples];
            copy_buffer.copy_from_slice(copy_stream);

            self.head = (self.head + n_samples) % buffer.len();
            self.count = std::cmp::min(self.count + n_samples, buffer.len());
        }
    }

    pub fn get_audio(&self, ms: usize, result: &mut Vec<T>) {
        let mut buffer = self.buffer.lock().unwrap();

        let mut n_samples = (ms as f64 * constants::SAMPLE_RATE / 1000f64) as usize;

        if n_samples > self.count {
            n_samples = self.count;
        }
        result.clear();
        result.resize(n_samples, T::default());

        // let mut out_buffer = vec![T::default(); n_samples as usize];

        let mut start_pos: i64 = self.head as i64 - n_samples as i64;

        if start_pos < 0 {
            start_pos += buffer.len() as i64;
        }

        let start_pos = start_pos as usize;

        if start_pos + n_samples > buffer.len() {
            let split_point = buffer.len() - start_pos;

            // let (out_left, out_right) = out_buffer.split_at_mut(split_point);
            let (out_left, out_right) = result.split_at_mut(split_point);

            out_left.copy_from_slice(&buffer[start_pos..=split_point]);
            out_right.copy_from_slice(&buffer[0..=(n_samples - split_point)]);
        } else {
            // out_buffer.copy_from_slice(&buffer[start_pos..=n_samples]);
            result.copy_from_slice(&buffer[start_pos..=n_samples]);
        }

        // self.sender.send(out_buffer).expect("failed to send data");
    }

    pub fn clear(&mut self) {
        let _buffer = self.buffer.lock().unwrap();
        self.head = 0;
        self.count = 0;
    }
}
