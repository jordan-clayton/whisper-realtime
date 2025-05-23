// TODO: Write a benchmark to test for each inner VAD implementation so that speed
// and accuracy claims can be made confidently.

use criterion::{Criterion, criterion_group, criterion_main};

// TODO: finish this once functions are in place to automate the boilerplate required to set up the transcriber
// Basic test plan:
// preload audio into shared buffer
// resample it to whisper sample-able and convert to chunks of constants::AUDIO_BUFFER_SIZE
// Pass a copy of the iterator to each of these vad functions

// In the VAD function, construct a RealtimeTranscriber
// Spawn scoped threads:
// One to write chunks from the iterator into the AudioRingBuffer
// One that's running the transcription thread
// One that's just draining the output buffer; these tests aren't concerned with accuracy.
// Let the threads run until all data has been broadcast and consumed (ie. channels all closed).

// Unwrap/Expect that all of these threads are going to terminate (they should not panic)
// They should also return successfully.
pub fn realtime_vad_benchmark(c: &mut Criterion) {
    c.bench_function("Silero realtime", |b| {
        b.iter(|| silero_realtime());
    });

    c.bench_function("Webrtc_realtime", |b| {
        b.iter(|| silero_realtime());
    });

    c.bench_function("Earshot", |b| {
        b.iter(|| silero_realtime());
    });
}

pub fn silero_realtime() {
    todo!()
}

pub fn webrtc_realtime() {
    todo!()
}
pub fn earshot_realtime() {
    todo!()
}

criterion_group!(benches, realtime_vad_benchmark);
criterion_main!(benches);
