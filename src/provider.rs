//! `TpuProvider`: a real device-discovery skeleton for Google Coral Edge
//! TPU devices via the real `libedgetpu` C API, deliberately not yet an
//! optimized, kernel-executing Provider.
//!
//! # Status: device discovery only, no compute kernels
//!
//! Unlike `providers/cuda`'s `CudaProvider` (a full, real-hardware-
//! verified Provider), this crate implements only the "does a real,
//! locally-attached Coral Edge TPU exist" half of that same pattern --
//! `TpuProvider::new()` always constructs successfully and gracefully
//! reports [`ProviderHealth::Unavailable`] when no Edge TPU runtime or
//! attached device is found, exactly like `CudaProvider` does for CUDA
//! and `RocmProvider`/`NpuProvider` do for ROCm/NPU, but it advertises
//! zero Kernels and zero real compute capability even when a device *is*
//! found. Real Kernel execution needs a real Coral Edge TPU to write and
//! verify against, which no machine this crate has been developed on
//! has. Implementing that is real, tracked future work for whoever has
//! that hardware -- not attempted here as unverifiable guesswork.
//!
//! # Graceful unavailability
//!
//! `TpuProvider::new()` never fails Runtime initialization for the
//! absence of hardware or the Edge TPU runtime shared library: it
//! attempts to dynamically load the real runtime library
//! (`edgetpu_sys.rs`'s `EDGETPU_LIBRARY_CANDIDATES`) and call
//! `edgetpu_list_devices`; any failure at any step (library not found,
//! symbol not found, or zero devices reported) is treated uniformly as
//! "no usable Edge TPU here" -- [`ProviderHealth::Unavailable`], not a
//! construction error. This is also what keeps this crate's `cargo test`
//! green on every machine and CI runner this repository currently has
//! (all genuinely without a Coral Edge TPU): the only thing that can fail
//! is the runtime `dlopen` attempt, which this module already treats as a
//! normal, expected outcome, the same posture `providers/rocm`/
//! `providers/npu` established first.

use magnetar_runtime::affinity::ProviderHealth;
use magnetar_runtime::provider::{Provider, ProviderError, ProviderMetadata, ProviderRegistry};

use crate::edgetpu_sys::{
    EDGETPU_LIBRARY_CANDIDATES, EdgetpuFreeDevicesFn, EdgetpuListDevicesFn,
};

const TPU_PROVIDER_NAME: &str = "tpu";
const TPU_PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");
const TPU_PROVIDER_VENDOR: &str = "Google Coral (Edge TPU)";

pub fn tpu_provider_metadata() -> ProviderMetadata {
    ProviderMetadata::new(
        TPU_PROVIDER_NAME,
        TPU_PROVIDER_VERSION,
        TPU_PROVIDER_VENDOR,
        "Device-discovery skeleton for Google Coral Edge TPU devices -- no \
         compute Kernels implemented yet (see this crate's own README)",
    )
}

/// The TPU Provider itself. Registers as a built-in Provider, matching
/// `RocmProvider`/`NpuProvider`/`CudaProvider`'s own posture (no
/// dynamic-library Provider ABI in this baseline).
pub struct TpuProvider {
    metadata: ProviderMetadata,
    /// `Some(count)` only when the real `libedgetpu` runtime was
    /// dynamically loaded and `edgetpu_list_devices` reported at least
    /// one attached device. `count` itself is currently unused beyond "is
    /// it positive" -- no per-device descriptor is built yet (see this
    /// module's own doc comment: device discovery only, not full Device
    /// enumeration).
    device_count: Option<usize>,
}

impl TpuProvider {
    pub fn new() -> Self {
        Self {
            metadata: tpu_provider_metadata(),
            device_count: Self::discover_device_count(),
        }
    }

    /// Attempts to dynamically load the real `libedgetpu` runtime library
    /// and query its real attached-device count. Returns `None` for *any*
    /// failure along the way (library not found, symbol not found, or
    /// zero devices) -- this baseline draws no distinction between those
    /// cases; they are all "no usable Edge TPU here" to a caller.
    fn discover_device_count() -> Option<usize> {
        let library = EDGETPU_LIBRARY_CANDIDATES
            .iter()
            .find_map(|name| unsafe { libloading::Library::new(name).ok() })?;

        // SAFETY: `edgetpu_list_devices`/`edgetpu_free_devices` are real,
        // stable, publicly documented Edge TPU C API entry points
        // (`edgetpu_sys.rs`, transcribed from the real `edgetpu_c.h`
        // header); the symbol lookups below only succeed against a real
        // `libedgetpu` runtime library actually exporting them under
        // these exact names, and the function-pointer/struct types
        // declared in `edgetpu_sys.rs` match the real C ABI. The returned
        // pointer is freed via `edgetpu_free_devices` before this
        // function returns, matching the real API's own documented
        // ownership contract (the caller owns, and must free, the
        // returned array).
        unsafe {
            let list_devices: libloading::Symbol<EdgetpuListDevicesFn> =
                library.get(b"edgetpu_list_devices\0").ok()?;
            let free_devices: libloading::Symbol<EdgetpuFreeDevicesFn> =
                library.get(b"edgetpu_free_devices\0").ok()?;

            let mut count: usize = 0;
            let devices = list_devices(&mut count);
            if !devices.is_null() {
                free_devices(devices);
            }
            if count == 0 { None } else { Some(count) }
        }
    }

    /// Whether this Provider found a usable, real, locally-attached Coral
    /// Edge TPU.
    pub fn is_available(&self) -> bool {
        self.device_count.is_some()
    }
}

impl Default for TpuProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for TpuProvider {
    fn metadata(&self) -> ProviderMetadata {
        self.metadata.clone()
    }

    fn register(&self, _registry: &mut ProviderRegistry) -> Result<(), ProviderError> {
        // No Devices to register yet -- this baseline does not build a
        // real `DeviceDescriptor` even when a device is found (see this
        // module's own "Status" doc comment); `devices()` keeps the
        // `Provider` trait's own empty-`Vec` default.
        Ok(())
    }

    fn health(&self) -> ProviderHealth {
        if self.is_available() {
            ProviderHealth::Available
        } else {
            ProviderHealth::Unavailable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one thing genuinely verified on every machine this crate has
    /// actually run on (this development environment, and CI's own
    /// Edge-TPU-less runners): a machine with no real Coral Edge TPU
    /// attached gets a Provider that constructs successfully and
    /// honestly reports itself unavailable, never a panic or a false
    /// "available".
    #[test]
    fn reports_unavailable_without_a_real_edge_tpu() {
        let provider = TpuProvider::new();
        assert!(
            !provider.is_available(),
            "this test environment has no real Coral Edge TPU attached"
        );
        assert_eq!(provider.health(), ProviderHealth::Unavailable);
    }

    #[test]
    fn metadata_is_well_formed() {
        let provider = TpuProvider::new();
        let metadata = provider.metadata();
        assert_eq!(metadata.name, TPU_PROVIDER_NAME);
        assert_eq!(metadata.vendor, TPU_PROVIDER_VENDOR);
        assert!(!metadata.description.is_empty());
    }

    #[test]
    fn devices_are_empty_without_a_real_edge_tpu() {
        let provider = TpuProvider::new();
        assert!(provider.devices().is_empty());
    }
}
