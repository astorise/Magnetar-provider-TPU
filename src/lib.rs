//! Google Coral Edge TPU execution Provider for the Magnetar local AI
//! Runtime.
//!
//! **Status: real device-discovery skeleton, no compute Kernels yet.**
//! See [`provider::TpuProvider`]'s own module doc comment and this
//! crate's README for what that means and why.

mod edgetpu_sys;
mod provider;

pub use provider::{TpuProvider, tpu_provider_metadata};
