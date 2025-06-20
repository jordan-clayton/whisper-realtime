use std::marker::PhantomData;

/// Encapsulates optional callbacks
pub trait Callback {
    type Argument;
    fn call(&mut self, arg: Self::Argument);
}

/// Encapsulates a basic FnMut callback for functions that accept optional callbacks.
#[repr(C)]
pub struct RibbleWhisperCallback<T, CB: FnMut(T)> {
    callback: CB,
    _marker: PhantomData<T>,
}
impl<T, CB: FnMut(T)> RibbleWhisperCallback<T, CB> {
    pub fn new(callback: CB) -> Self {
        Self {
            callback,
            _marker: PhantomData,
        }
    }
}

impl<T, CB: FnMut(T)> Callback for RibbleWhisperCallback<T, CB> {
    type Argument = T;
    fn call(&mut self, arg: T) {
        (&mut self.callback)(arg);
    }
}

/// This is the static equivalent of  [RibbleWhisperCallback]
/// Encouraged for use when `'static` lifetimes are required, (e.g. OfflineWhisperProgressCallback).
/// It is not strictly necessary to use this over ProgressCallback, but it may help with
/// locating and debugging lifetime errors.
#[repr(C)]
pub struct StaticRibbleWhisperCallback<T, CB: FnMut(T) + 'static> {
    callback: CB,
    _marker: PhantomData<T>,
}

impl<T, CB: FnMut(T) + 'static> StaticRibbleWhisperCallback<T, CB> {
    pub fn new(callback: CB) -> Self {
        Self {
            callback,
            _marker: PhantomData,
        }
    }
}

impl<T, CB: FnMut(T) + 'static> Callback for StaticRibbleWhisperCallback<T, CB> {
    type Argument = T;
    fn call(&mut self, arg: T) {
        (&mut self.callback)(arg);
    }
}

/// To indicate "None" in functions that expect a callback (eg. downloading, loading audio, etc.)
/// Since all optional callbacks are unpacked before a hot loop to cut down on branching,
/// this will get called repeatedly but expect this to be optimized out.
///
/// This can also be used with Option<impl Callback> = None to satisfy type annotation requirements.
#[repr(C)]
pub struct Nop<T> {
    _marker: PhantomData<T>,
}

impl<T> Nop<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<T> Callback for Nop<T> {
    type Argument = T;
    fn call(&mut self, _arg: T) {}
}
