use std::marker::PhantomData;

pub trait Callback {
    type Argument;
    fn call(&mut self, arg: Self::Argument);
}

/// Basic callback struct to encapsulate an FnMut closure
/// Use if you require to supply a callback explicitly, otherwise, just use the API.
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

/// This is to deal with optional progress callbacks without repeated branching in hot loops.
/// When the callback is unnecessary, supply NOP
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
