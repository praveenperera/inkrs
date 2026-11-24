//! inkrs: GPU-resident inference for villa's ink and surface models: the flat ink readers (`hybrid_3d2d-seed42`,
//! `reader-v2`), Hecate 9.6 um, the v8-in family, the 3D ink U-Net and the nnU-Net surface model; the commands live
//! in `inkrs::cli`

use color_eyre::eyre::Result;

// macOS malloc keeps freed large blocks resident; this maps them instead, so a model's freed load copies leave the
// footprint (src/metal/alloc.rs)
#[cfg(target_os = "macos")]
#[global_allocator]
static ALLOC: inkrs::metal::alloc::LargePages = inkrs::metal::alloc::LargePages;

fn main() -> Result<()> {
    // Windows: before any library loads, so that a load by bare name after the DLL checks never searches the current
    // directory or PATH (src/dll.rs)
    #[cfg(windows)]
    inkrs::dll::restrict_search()?;

    // the driver JIT cache is shared by every CUDA program and evicts its oldest entries at 1 GiB
    // (a busy box sits at that cap), so other programs can evict the fused convs' compiled kernels
    // and bring their first-use compile back; ours goes in the inkrs cache. Set here, before any
    // CUDA call, since the driver reads it once at initialisation. The Python module never sets it, since
    // it may share a process with torch, whose CUDA is already initialised; the wheel's console script,
    // which starts its own process, sets it as this does
    #[cfg(not(target_os = "macos"))]
    if let Some(path) = inkrs::dnn::algo_cache::cuda_cache_path(&|name| std::env::var_os(name)) {
        // the driver's handling of a missing directory is not documented; a failure here leaves it
        // to the driver
        let _ = std::fs::create_dir_all(&path);
        // SAFETY: the start of main, before any other thread exists
        unsafe { std::env::set_var("CUDA_CACHE_PATH", path) };
    }
    color_eyre::install()?;
    inkrs::cli::run(std::env::args_os())
}
