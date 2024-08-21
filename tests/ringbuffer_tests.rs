#[cfg(test)]
mod ringbuffer_tests {
    use std::sync::atomic::Ordering;

    use whisper_realtime::{audio_ring_buffer::AudioRingBuffer, constants};

    #[test]
    fn test_copy_buffer_lengths() {
        // Full length
        let mut ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new(constants::SAMPLE_DURATION);
        let sample_len =
            (constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&mut samples);
        assert_eq!(ring_buffer.audio_len.load(Ordering::Acquire), sample_len);

        // Half-length
        let mut ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new(constants::SAMPLE_DURATION);
        let sample_len =
            (constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE / 2f64;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&mut samples);
        assert_eq!(ring_buffer.audio_len.load(Ordering::Acquire), sample_len);
        assert_eq!(
            ring_buffer.buffer_len.load(Ordering::Acquire),
            sample_len * 2
        );
    }

    #[test]
    fn test_insert_and_get() {
        let mut ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new(constants::SAMPLE_DURATION);

        // Insert four seconds of audio data.
        let sample_len = 4f64 * constants::WHISPER_SAMPLE_RATE;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&mut samples);

        // Ensure that 4s of audio data are in the buffer.
        assert_eq!(ring_buffer.audio_len.load(Ordering::Acquire), sample_len);

        // Get two seconds of audio data.
        let mut result: Vec<f32> = vec![];
        // This is expected to not panic
        ring_buffer.get_audio(2000, &mut result);
        // Test length
        assert_eq!(result.len(), sample_len / 2);

        // Ensure all 0.5
        let right_data = &result.iter().all(|x| *x >= 0.49f32 && *x <= 0.51f32);
        assert!(right_data);
    }
    #[test]
    fn test_insert_and_overflow() {
        // Half-length
        let mut ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new(constants::SAMPLE_DURATION);
        let sample_len =
            (constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE / 2f64;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];

        // This is expected to not panic.
        for i in 0..3 {
            ring_buffer.push_audio(&mut samples);
        }

        // Expect a full buffer
        assert_eq!(
            ring_buffer.audio_len.load(Ordering::Acquire),
            ring_buffer.buffer_len.load(Ordering::Acquire)
        );
        // Expect head to be at the midpoint.
        assert_eq!(
            ring_buffer.get_head_position(),
            ring_buffer.audio_len.load(Ordering::Acquire) / 2
        );
    }

    #[test]
    fn test_get_with_overflow() {
        // Half-length
        let mut ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new(constants::SAMPLE_DURATION);
        let sample_len =
            (constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE / 2f64;
        let sample_len = sample_len as usize;

        let expect_length = (2f64 * constants::WHISPER_SAMPLE_RATE) as usize;

        let mut results_vec: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        // This is expected to not panic.
        for i in 0..3 {
            let mut samples = vec![i as f32; sample_len];
            // Insert 5s of samples.
            ring_buffer.push_audio(&mut samples);
            // Get 2s of samples.
            ring_buffer.get_audio(2000, &mut results_vec[i]);
        }

        // Length check.
        for vec in results_vec {
            assert_eq!(vec.len(), expect_length);
            let nd = non_decreasing(&vec);
            assert!(nd);
        }
    }

    fn non_decreasing(v: &Vec<f32>) -> bool {
        for i in 0..v.len() - 1 {
            let j = i + 1;
            if v[j] < v[i] {
                return false;
            }
        }
        true
    }
}
