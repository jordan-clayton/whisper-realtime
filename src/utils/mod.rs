pub mod callback;
pub mod constants;
pub mod errors;

/// Type alias to handle channel configurations
#[cfg(not(feature = "crossbeam"))]
pub type Receiver<T> = std::sync::mpsc::Receiver<T>;
#[cfg(not(feature = "crossbeam"))]
pub type Sender<T> = std::sync::mpsc::SyncSender<T>;
#[cfg(feature = "crossbeam")]
pub type Sender<T> = crossbeam::channel::Sender<T>;
#[cfg(feature = "crossbeam")]
pub type Receiver<T> = crossbeam::channel::Receiver<T>;

/// Returns the appropriate channel type based on enabled features (crossbeam)
/// Used for passing audio and text while transcribing
pub fn get_channel<T>(channel_size: usize) -> (Sender<T>, Receiver<T>) {
    #[cfg(not(feature = "crossbeam"))]
    {
        std::sync::mpsc::sync_channel(channel_size)
    }
    #[cfg(feature = "crossbeam")]
    {
        crossbeam::channel::bounded(channel_size)
    }
}
