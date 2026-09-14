# magnetar-provider-tpu

## Purpose

A Google Coral Edge TPU execution Provider for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime,
implementing Magnetar's `Provider` contract for Edge TPU Devices via the
real, public `libedgetpu` C API.

**"TPU" here means the real, purchasable local Coral Edge TPU accelerator**
(a USB or PCIe device) -- **not Google Cloud TPU**, which is a cloud
service with no local device to discover and out of scope for a
local-device-discovery Provider by construction.

## Status

**Real device-discovery skeleton, not a full Provider.** `TpuProvider::new()`
dynamically loads the real Edge TPU runtime shared library (`edgetpu.dll`
on Windows, `libedgetpu.so.1` on Linux, `libedgetpu.1.dylib` on macOS -- no
build-time dependency on any of them, the same `libloading`-based
dynamic-loading posture
[`providers/rocm`](https://github.com/astorise/Magnetar-provider-ROCm)/
[`providers/npu`](https://github.com/astorise/Magnetar-provider-NPU)
already establish) and calls the real, public `edgetpu_list_devices` API
entry point (transcribed directly from the real, public `edgetpu_c.h`
header, not recalled from memory) to discover whether a real, locally
attached Coral Edge TPU exists. When it does not -- including on every
machine and CI runner this crate has actually been developed and tested
on, none of which has a real Coral device attached -- it gracefully
reports `ProviderHealth::Unavailable` and zero Devices, exactly like
`CudaProvider` does for CUDA and `RocmProvider`/`NpuProvider` do for
ROCm/NPU when no compatible hardware is present.

**What is real and verified**: the graceful-unavailability path itself.
`TpuProvider::new()` constructs successfully and `is_available()`/
`health()` correctly report "no Edge TPU here" on a genuinely
Edge-TPU-less machine -- confirmed on this crate's own development
environment and confirmed again by this repository's CI (`ubuntu-latest`,
also genuinely without a Coral device), the same two-environments-agree
verification `providers/rocm`'s own README already establishes for the
equivalent ROCm case.

**What is explicitly not implemented, and why**: no compute Kernels
(`ProviderExecutionApi`), no per-device enumeration beyond a raw count, no
Kernel advertisements. Writing real Edge TPU compute kernels (which, for
this hardware, means compiling a model into Edge TPU's own custom
instruction format ahead of time via the Edge TPU Compiler -- a
fundamentally different execution model from CUDA/ROCm/NPU's runtime
kernel dispatch) and claiming they are correct without ever running them
on a real Coral device would be unverifiable guesswork -- exactly the
category of confident-but-untested work this repository's own development
practice avoids everywhere else. This crate stops at the boundary of what
can actually be verified from an Edge-TPU-less development environment:
real device discovery, honestly reporting unavailability when that is the
truth.

## What a future contributor with a real Coral Edge TPU would need to do

1. Extend `src/edgetpu_sys.rs` with real per-device path/type reporting
   (the `edgetpu_device` struct this baseline already declares but does
   not yet surface per-device), building real
   `magnetar_runtime::device::DeviceDescriptor` values from it.
2. Design how Edge TPU's ahead-of-time-compiled execution model (via the
   separate Edge TPU Compiler toolchain, not a runtime kernel-source
   compilation API like CUDA/HIP/Level Zero) fits Magnetar's
   `ProviderExecutionApi`/Kernel Registry dispatch contract -- a real,
   nontrivial design question this baseline does not attempt to answer
   speculatively.
3. Verify against `providers/cpu`'s reference implementation on real Coral
   hardware, the same `provider-compute` conformance profile
   `providers/cuda` already passes for CUDA.

## Governing contract

Implements Magnetar's generic `Provider`/`Device`/`ProviderExecutionApi`
contracts from the main [Magnetar](https://github.com/astorise/Magnetar)
repository's `magnetar-runtime` crate. No dedicated OpenSpec capability
exists yet for this crate specifically; one should be scoped in the main
repository's `openspec/specs/` once real Kernel execution work here
begins, the same way `providers/cuda`'s own `cuda-provider` capability was.

## Relationship to magnetar-runtime

This Provider is loaded and driven by the Runtime's own Provider registry,
never the reverse -- `magnetar-runtime` has zero compile-time dependency
on this crate (the same externalization invariant every Provider/Component/
Format module in this workspace observes). It is pinned into the main
Magnetar repository as a git submodule at `providers/tpu`.
