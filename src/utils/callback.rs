use std::marker::PhantomData;

/// Encapsulates optional callbacks
pub trait Callback {
    type Argument;
    fn call(&mut self, arg: Self::Argument);
}

/// Encapsulates progress-related callbacks
#[repr(C)]
pub struct ProgressCallback<T, CB: FnMut(T)> {
    callback: CB,
    _marker: PhantomData<T>,
}
impl<T, CB: FnMut(T)> ProgressCallback<T, CB> {
    pub fn new(callback: CB) -> Self {
        Self {
            callback,
            _marker: PhantomData,
        }
    }
}

impl<T, CB: FnMut(T)> Callback for ProgressCallback<T, CB> {
    type Argument = T;
    fn call(&mut self, arg: T) {
        (&mut self.callback)(arg);
    }
}

/// This is the static equivalent of  [crate::utils::callback::ProgressCallback]
/// Encouraged for use when 'static lifetimes are required, (eg. OfflineWhisperProgressCallback).
/// It is not strictly necessary to use this over ProgressCallback, but it may help with
/// locating and debugging lifetime errors.
#[repr(C)]
pub struct StaticProgressCallback<T, CB: FnMut(T) + 'static> {
    callback: CB,
    _marker: PhantomData<T>,
}

impl<T, CB: FnMut(T) + 'static> StaticProgressCallback<T, CB> {
    pub fn new(callback: CB) -> Self {
        Self {
            callback,
            _marker: PhantomData,
        }
    }
}

impl<T, CB: FnMut(T) + 'static> Callback for StaticProgressCallback<T, CB> {
    type Argument = T;
    fn call(&mut self, arg: T) {
        (&mut self.callback)(arg);
    }
}

/// To indicate "None" in functions that expect a callback (eg. downloading, loading audio, etc.)
/// Since all optional callbacks are called repeatedly
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
