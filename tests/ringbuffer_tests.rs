#[cfg(test)]
mod ringbuffer_tests {
    use whisper_realtime::audio::audio_ring_buffer::AudioRingBuffer;
    use whisper_realtime::utils::constants;

    #[test]
    fn test_copy_buffer_lengths() {
        // Full length
        let ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new_with_size(constants::SAMPLE_DURATION, None)
                .expect("Audio length needs to be non-zero");
        let sample_len =
            (constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&mut samples);
        assert_eq!(ring_buffer.audio_len(), sample_len);

        // Half-length
        let ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new_with_size(constants::SAMPLE_DURATION, None)
                .expect("Audio length needs to be non-zero");
        let sample_len =
            (constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE / 2f64;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&mut samples);
        assert_eq!(ring_buffer.audio_len(), sample_len);
        assert_eq!(ring_buffer.buffer_len(), sample_len * 2);
    }

    #[test]
    fn test_insert_and_get() {
        let ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new_with_size(constants::SAMPLE_DURATION, None)
                .expect("Audio length should be non-zero");

        // Insert four seconds of audio data.
        let sample_len = 4f64 * constants::WHISPER_SAMPLE_RATE;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];
        // This is expected to not panic.
        ring_buffer.push_audio(&mut samples);

        // Ensure that 4s of audio data are in the buffer.
        assert_eq!(ring_buffer.audio_len(), sample_len);

        // Get two seconds of audio data.
        let mut result: Vec<f32> = vec![];
        // This is expected to not panic
        ring_buffer.read_into(2000, &mut result);
        // Test length
        assert_eq!(result.len(), sample_len / 2);

        // Ensure all 0.5
        let right_data = &result.iter().all(|x| *x >= 0.49f32 && *x <= 0.51f32);
        assert!(right_data);
    }

    #[test]
    fn test_insert_and_overflow() {
        // Half-length
        let ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new_with_size(constants::SAMPLE_DURATION, None)
                .expect("Audio length needs to be non-zero");
        let sample_len =
            (constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE / 2f64;
        let sample_len = sample_len as usize;

        let mut samples = vec![0.5f32; sample_len];

        // This is expected to not panic.
        for _i in 0..3 {
            ring_buffer.push_audio(&mut samples);
        }

        // Expect a full buffer
        assert_eq!(ring_buffer.audio_len(), ring_buffer.buffer_len());
        // Expect head to be at the midpoint.
        assert_eq!(ring_buffer.get_head_position(), ring_buffer.audio_len() / 2);
    }
    #[test]
    fn test_wraparound_audio() {
        // Half-length
        let ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new_with_size(constants::SAMPLE_DURATION, None)
                .expect("Audio length should be non-zero");
        // These are 5-second samples
        let sample_len =
            ((constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE) / 2f64;
        let sample_len = sample_len as usize;

        // Fill the first half with 0.1
        let first_half_samples = vec![0.1f32; sample_len];
        // Fill the second half with 0.2
        let second_half_samples = vec![0.2f32; sample_len];

        // This is expected to not panic.
        ring_buffer.push_audio(&first_half_samples);
        ring_buffer.push_audio(&second_half_samples);
        // Move the head to position 1/4
        ring_buffer.push_audio(&first_half_samples[0..sample_len / 2]);

        // Expect a full buffer
        assert_eq!(ring_buffer.audio_len(), ring_buffer.buffer_len());
        // Expect head to be a quarter way in.
        assert_eq!(ring_buffer.get_head_position(), ring_buffer.audio_len() / 4);

        // Read 5 seconds of audio
        let mut read_buffer = vec![];
        ring_buffer.read_into(constants::SAMPLE_DURATION / 2, &mut read_buffer);
        // Bisect at the midpoint and compare the audio output
        let (first_half, second_half) = read_buffer.split_at(sample_len / 2);
        assert_eq!(first_half.len(), second_half.len());

        let first_wrong = first_half.iter().position(|s| *s != 0.2);
        let second_wrong = second_half.iter().position(|s| *s != 0.1);

        assert!(
            first_wrong.is_none(),
            "Found 0.1 in first half at index: {}",
            first_wrong.unwrap()
        );
        assert!(
            second_wrong.is_none(),
            "Found 0.2 in second half at index: {}",
            second_wrong.unwrap()
        );
    }

    #[test]
    fn test_get_with_overflow() {
        // Half-length
        let ring_buffer: AudioRingBuffer<f32> =
            AudioRingBuffer::new_with_size(constants::SAMPLE_DURATION, None)
                .expect("Audio length should be non-zero");
        let sample_len =
            ((constants::SAMPLE_DURATION / 1000) as f64 * constants::WHISPER_SAMPLE_RATE) / 2f64;
        let sample_len = sample_len as usize;

        let expect_length = (2f64 * constants::WHISPER_SAMPLE_RATE) as usize;

        let mut results_vec: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        // This is expected to not panic.
        for i in 0..3 {
            let mut samples = vec![i as f32; sample_len];
            // Insert 5s of samples.
            ring_buffer.push_audio(&mut samples);
            // Get 2s of samples.
            ring_buffer.read_into(2000, &mut results_vec[i]);
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
