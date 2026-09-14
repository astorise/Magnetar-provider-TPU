//! Minimal, hand-declared FFI surface for Google Coral's real Edge TPU
//! runtime C API -- only the one entry point this baseline's device
//! discovery actually calls, matching the real public
//! `libedgetpu`/`edgetpu_c.h` API declarations, transcribed directly from
//! the real, public `google-coral/libedgetpu` header (not recalled from
//! memory): `edgetpu_list_devices`.
//!
//! "TPU" here means the real, purchasable local Coral Edge TPU
//! accelerator (a USB or PCIe device) -- not Google Cloud TPU, a cloud
//! service with no local device to discover, and out of scope for a
//! local-device-discovery Provider by construction.
//!
//! # Honesty about verification
//!
//! These declarations are transcribed verbatim from the real, public
//! `edgetpu_c.h` header (`google-coral/libedgetpu`, fetched directly
//! rather than recalled from memory) and are exercised for real by
//! [`crate::provider::TpuProvider::new`]'s device-discovery path on every
//! machine this repository's own tests and CI actually run on -- but no
//! machine this crate has been developed or tested on has a real Coral
//! Edge TPU or `libedgetpu` runtime installed. The only path genuinely
//! exercised so far is the runtime library's absence (dynamic loading
//! fails outright) -- never a real, successful call against real
//! hardware. See this crate's own README for what would be needed to
//! close that gap.

use std::os::raw::{c_char, c_int};

/// `enum edgetpu_device_type` -- real values from `edgetpu_c.h`. Not
/// currently interpreted by this baseline beyond counting the devices
/// returned; kept as a real, correctly-shaped field rather than omitted,
/// since the struct's layout must match the real C ABI exactly for
/// `edgetpu_list_devices`' return value to be read safely at all.
#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgetpuDeviceType {
    ApexPci = 0,
    ApexUsb = 1,
}

/// `struct edgetpu_device { enum edgetpu_device_type type; const char* path; };`
/// -- real layout from `edgetpu_c.h`, `#[repr(C)]` to match the real C
/// struct layout `edgetpu_list_devices` actually returns an array of.
#[repr(C)]
pub struct EdgetpuDevice {
    pub device_type: c_int,
    pub path: *const c_char,
}

/// `struct edgetpu_device* edgetpu_list_devices(size_t* num_devices)` --
/// real Edge TPU C API. Returns a pointer to a real, runtime-allocated
/// array of `num_devices` real `EdgetpuDevice` entries (or a null/empty
/// result when no device is attached); must be freed via
/// `edgetpu_free_devices`, which this baseline's discovery path does
/// immediately after reading the count.
pub type EdgetpuListDevicesFn = unsafe extern "C" fn(num_devices: *mut usize) -> *mut EdgetpuDevice;

/// `void edgetpu_free_devices(struct edgetpu_device* dev)`.
pub type EdgetpuFreeDevicesFn = unsafe extern "C" fn(dev: *mut EdgetpuDevice);

/// Real Edge TPU runtime shared library file names this crate attempts to
/// dynamically load, in order, per platform -- the real, documented
/// install-time artifact names a real `libedgetpu` installation provides.
/// No build-time dependency on any of them (`libloading` resolves purely
/// at runtime), matching `providers/rocm`/`providers/npu`'s own
/// dynamic-loading posture.
#[cfg(target_os = "windows")]
pub const EDGETPU_LIBRARY_CANDIDATES: &[&str] = &["edgetpu.dll"];
#[cfg(target_os = "linux")]
pub const EDGETPU_LIBRARY_CANDIDATES: &[&str] = &["libedgetpu.so.1", "libedgetpu.so"];
#[cfg(target_os = "macos")]
pub const EDGETPU_LIBRARY_CANDIDATES: &[&str] = &["libedgetpu.1.dylib"];
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
pub const EDGETPU_LIBRARY_CANDIDATES: &[&str] = &[];
