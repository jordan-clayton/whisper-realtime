use std::io::{self, Read};

use super::downloader;

pub struct DownloadProgress<R: Read, CB: downloader::DownloadCallback> {
    file_stream: R,
    // NOTE: These might be redundant with the use of a callback.
    // TODO: Refactor accordingly.
    progress: usize,
    total_size: usize,
    update_progress: CB,
}

impl<R, CB> DownloadProgress<R, CB> {
    fn new(file_stream: R, total_size: usize, update_progress: CB) -> DownloadProgress<R, CB> {
        DownloadProgress {
            file_stream,
            progress: 0,
            total_size,
            update_progress,
        }
    }
}

// TODO: error handling for a bad read.
impl<R: Read, CB> Read for DownloadProgress<R, CB> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.file_stream.read(buf);
        if let Ok(num_bytes) = read {
            self.progress += num_bytes;
            self.update_progress(num_bytes);
        } else {
            self.update_progress(0);
        }
        read
    }
}
