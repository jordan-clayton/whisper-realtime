#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::sync_channel;
use std::thread::{scope, sleep};
use std::time::Duration;

use criterion::{black_box, Criterion, criterion_group, criterion_main};
#[cfg(feature = "crossbeam")]
use crossbeam::channel;
use sdl2::audio::AudioCallback;

use whisper_realtime::constants;

// Benchmark summary:
// Arc<[T]> is orders of magnitude faster in single thread, multi-thread with light work
// Vec<T> in parallel is, abnormally, reasonably fast.

// Regardless: The bottleneck comes down to worker threads, it doesn't make a significant difference
// to prefer slices over Vec<T>

pub fn recorder_benchmark(c: &mut Criterion) {
    // Single thread: Sending data only
    let mut group = c.benchmark_group("Single thread: sending data only");
    group.bench_function("recorder_st: 50 audio samples", |b| {
        b.iter(|| run_with_recorder_st(black_box(50)))
    });

    group.bench_function("slice_st: 50 audio samples", |b| {
        b.iter(|| run_with_slice_st(black_box(50)))
    });

    group.finish();

    // Multi thread: Sending and receiving data, no work is done
    let mut group = c.benchmark_group("Multi thread: sending and receiving data, no work is done");
    group.bench_function("recorder_par: 50 audio samples", |b| {
        b.iter(|| run_with_recorder_par(black_box(50), black_box(0)))
    });

    group.bench_function("slice_par: 50 audio samples", |b| {
        b.iter(|| run_with_slice_par(black_box(50), black_box(0)))
    });

    group.finish();

    // Multi thread: Sending and receiving data, work is simulated
    let mut group = c
        .benchmark_group("Multi thread: sending and receiving data, work is simulated using sleep");

    group.bench_function("recorder_par_with_workers: 5 audio samples", |b| {
        b.iter(|| run_with_recorder_par(black_box(5), black_box(10)))
    });

    group.bench_function("slice_par_with_workers: 5 samples", |b| {
        b.iter(|| run_with_slice_par(black_box(5), black_box(10)))
    });

    group.finish();
}

fn run_with_recorder_st(n_samples: usize) {
    let sample_size = constants::WHISPER_SAMPLE_RATE as usize;
    let audio_len = n_samples * sample_size;
    let mut audio = vec![0.0f32; audio_len];
    #[cfg(feature = "crossbeam")]
    let (a_sender, a_receiver) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (a_sender, a_receiver) = sync_channel(n_samples + 1);
    let mut recorder = whisper_realtime::recorder::Recorder { sender: a_sender };

    // Prepare channels for multiple worker threads to operate on the data simultaneously
    #[cfg(feature = "crossbeam")]
    let (wt1_sender, _wt1_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt1_sender, _wt1_recv) = sync_channel(n_samples + 1);
    #[cfg(feature = "crossbeam")]
    let (wt2_sender, _wt2_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt2_sender, _wt2_recv) = sync_channel(n_samples + 1);
    #[cfg(feature = "crossbeam")]
    let (wt3_sender, _wt3_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt3_sender, _wt3_recv) = sync_channel(n_samples + 1);

    // Fill the audio channel with n_samples
    for sample in audio.chunks_mut(sample_size) {
        recorder.callback(sample)
    }

    // Drain the audio sending channel and send out to multiple worker channels
    while let Ok(sample) = a_receiver.try_recv() {
        wt1_sender
            .send(sample.clone())
            .expect("Failed to send vector sample to wt1");
        wt2_sender
            .send(sample.clone())
            .expect("Failed to send vector sample to wt2");
        wt3_sender
            .send(sample.clone())
            .expect("Failed to send vector sample to wt3");
    }
}

fn run_with_slice_st(n_samples: usize) {
    let sample_size = constants::WHISPER_SAMPLE_RATE as usize;
    let audio_len = n_samples * sample_size;
    let mut audio = vec![0.0f32; audio_len];
    #[cfg(feature = "crossbeam")]
    let (a_sender, a_receiver) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (a_sender, a_receiver) = sync_channel(n_samples + 1);
    let mut recorder = whisper_realtime::recorder::SliceRecorder { sender: a_sender };

    // Prepare channels for multiple worker threads to operate on the data simultaneously
    #[cfg(feature = "crossbeam")]
    let (wt1_sender, _wt1_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt1_sender, _wt1_recv) = sync_channel(n_samples + 1);
    #[cfg(feature = "crossbeam")]
    let (wt2_sender, _wt2_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt2_sender, _wt2_recv) = sync_channel(n_samples + 1);
    #[cfg(feature = "crossbeam")]
    let (wt3_sender, _wt3_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt3_sender, _wt3_recv) = sync_channel(n_samples + 1);

    // Fill the audio channel with n_samples
    for sample in audio.chunks_mut(sample_size) {
        recorder.callback(sample)
    }

    // Drain the audio sending channel and send out to multiple worker channels
    while let Ok(sample) = a_receiver.try_recv() {
        wt1_sender
            .send(sample.clone())
            .expect("Failed to send slice sample to wt1");
        wt2_sender
            .send(sample.clone())
            .expect("Failed to send slice sample to wt2");
        wt3_sender
            .send(sample.clone())
            .expect("Failed to send slice sample to wt3");
    }
}

fn run_with_recorder_par(n_samples: usize, work_millis: u64) {
    let sample_size = constants::WHISPER_SAMPLE_RATE as usize;
    let audio_len = n_samples * sample_size;
    let mut audio = vec![0.0f32; audio_len];
    #[cfg(feature = "crossbeam")]
    let (a_sender, a_receiver) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (a_sender, a_receiver) = sync_channel(n_samples + 1);

    let mut recorder = whisper_realtime::recorder::Recorder { sender: a_sender };

    // Prepare channels for multiple worker threads to operate on the data simultaneously
    #[cfg(feature = "crossbeam")]
    let (wt1_sender, wt1_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt1_sender, wt1_recv) = sync_channel(n_samples + 1);
    #[cfg(feature = "crossbeam")]
    let (wt2_sender, wt2_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt2_sender, wt2_recv) = sync_channel(n_samples + 1);
    #[cfg(feature = "crossbeam")]
    let (wt3_sender, wt3_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt3_sender, wt3_recv) = sync_channel(n_samples + 1);

    scope(|s| {
        let _send = s.spawn(move || {
            // Fill the audio channel with n_samples
            for sample in audio.chunks_mut(sample_size) {
                recorder.callback(sample)
            }
            drop(recorder);
        });
        let _read = s.spawn(move || {
            loop {
                let output = a_receiver.recv();
                match output {
                    Ok(audio) => {
                        wt1_sender
                            .send(audio.clone())
                            .expect("Failed to send vector to thread 1");
                        wt2_sender
                            .send(audio.clone())
                            .expect("Failed to send vector to thread 2");
                        wt3_sender
                            .send(audio.clone())
                            .expect("Failed to send vector to thread 3");
                    }
                    // Break on no more senders;
                    // The recorder will go out of scope when _send finishes, and so will the sender
                    Err(_) => break,
                }
            }
            drop(wt1_sender);
            drop(wt2_sender);
            drop(wt3_sender);
        });
        // Worker threads to simulate working on audio
        // Suppose wt1 is a write thread and takes slightly longer due to IO
        let _wt1 = s.spawn(move || loop {
            let output = wt1_recv.recv();
            match output {
                Ok(_) => {
                    if work_millis > 0 {
                        sleep(Duration::from_millis(work_millis));
                    }
                }
                Err(_) => break,
            }
        });
        let _wt2 = s.spawn(move || loop {
            let output = wt2_recv.recv();
            match output {
                Ok(_) => {
                    let frac_millis = work_millis / 5;
                    if frac_millis > 0 {
                        sleep(Duration::from_millis(frac_millis));
                    }
                }
                Err(_) => break,
            }
        });

        let _wt3 = s.spawn(move || loop {
            let output = wt3_recv.recv();
            match output {
                Ok(_) => {
                    let frac_millis = work_millis / 5;
                    if frac_millis > 0 {
                        sleep(Duration::from_millis(frac_millis));
                    }
                }
                Err(_) => break,
            }
        });
    });
}
fn run_with_slice_par(n_samples: usize, work_millis: u64) {
    let sample_size = constants::WHISPER_SAMPLE_RATE as usize;
    let audio_len = n_samples * sample_size;
    let mut audio = vec![0.0f32; audio_len];
    #[cfg(feature = "crossbeam")]
    let (a_sender, a_receiver) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (a_sender, a_receiver) = sync_channel(n_samples + 1);

    let mut recorder = whisper_realtime::recorder::SliceRecorder { sender: a_sender };

    // Prepare channels for multiple worker threads to operate on the data simultaneously
    #[cfg(feature = "crossbeam")]
    let (wt1_sender, wt1_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt1_sender, wt1_recv) = sync_channel(n_samples + 1);
    #[cfg(feature = "crossbeam")]
    let (wt2_sender, wt2_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt2_sender, wt2_recv) = sync_channel(n_samples + 1);
    #[cfg(feature = "crossbeam")]
    let (wt3_sender, wt3_recv) = channel::bounded(n_samples + 1);
    #[cfg(not(feature = "crossbeam"))]
    let (wt3_sender, wt3_recv) = sync_channel(n_samples + 1);

    scope(|s| {
        let _send = s.spawn(move || {
            // Fill the audio channel with n_samples
            // When the loop finishes, the recorder should go out of scope and be dropped
            for sample in audio.chunks_mut(sample_size) {
                recorder.callback(sample)
            }
            drop(recorder);
        });
        let _read = s.spawn(move || {
            loop {
                let output = a_receiver.recv();
                match output {
                    Ok(audio) => {
                        wt1_sender
                            .send(audio.clone())
                            .expect("Failed to send slice to thread 1");
                        wt2_sender
                            .send(audio.clone())
                            .expect("Failed to send slice to thread 2");
                        wt3_sender
                            .send(audio.clone())
                            .expect("Failed to send slice to thread 3");
                    }
                    // Break on no more senders;
                    // The recorder will go out of scope when _send finishes, and so will the sender
                    Err(_) => break,
                }
            }
            drop(wt1_sender);
            drop(wt2_sender);
            drop(wt3_sender);
        });
        // Worker threads to simulate working on audio
        // Suppose wt1 is a write thread and takes slightly longer due to IO
        let _wt1 = s.spawn(move || loop {
            let output = wt1_recv.recv();
            match output {
                Ok(_) => {
                    if work_millis > 0 {
                        sleep(Duration::from_millis(work_millis));
                    }
                }
                Err(_) => break,
            }
        });
        let _wt2 = s.spawn(move || loop {
            let output = wt2_recv.recv();
            match output {
                Ok(_) => {
                    let frac_millis = work_millis / 5;
                    if frac_millis > 0 {
                        sleep(Duration::from_millis(frac_millis));
                    }
                }
                Err(_) => break,
            }
        });

        let _wt3 = s.spawn(move || loop {
            let output = wt3_recv.recv();
            match output {
                Ok(_) => {
                    let frac_millis = work_millis / 5;
                    if frac_millis > 0 {
                        sleep(Duration::from_millis(frac_millis));
                    }
                }
                Err(_) => break,
            }
        });
    });
}

criterion_group!(benches, recorder_benchmark);
criterion_main!(benches);
