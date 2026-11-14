//! `inkrs-afv-trace`: a native, multi-threaded drop-in for `avf-spline-generator` 0.2.0
//! (Qualzz/AVF_SPLINE_GENERATOR @ bee798f7, Apache-2.0) that writes the same `.afv`

mod afv;
mod axes;
mod fields;
mod finish;
mod input;
mod kernels;
mod pyfmt;
mod run;
mod vol;

use std::ffi::OsString;

use clap::{CommandFactory, Parser};

/// Print one log line as the reference's `logging` setup does: `HH:MM:SS message` on stderr
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        eprintln!("{} {}", $crate::clock(), format_args!($($arg)*))
    };
}

/// Local wall-clock time as `%H:%M:%S`
pub fn clock() -> String {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let now = unsafe { libc::time(std::ptr::null_mut()) };
    unsafe { libc::localtime_r(&now, &mut tm) };
    format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
}

extern "C" fn on_interrupt(_: libc::c_int) {
    // only async-signal-safe calls here: write(2) and _exit(2)
    const MSG: &[u8] = b"Interrupted. Run the same command with --resume to continue.\n";
    unsafe {
        libc::write(2, MSG.as_ptr().cast(), MSG.len());
        libc::_exit(130);
    }
}

fn main() {
    // numpy's OpenBLAS is loaded only for 3x3 eigenproblems; its own thread pool would only add threads
    // beyond --threads. Set before any thread exists
    unsafe { std::env::set_var("OPENBLAS_NUM_THREADS", "1") };
    unsafe { libc::signal(libc::SIGINT, on_interrupt as extern "C" fn(libc::c_int) as libc::sighandler_t) };
    let argv: Vec<OsString> = std::env::args_os().collect();
    // argparse runs the help and version actions as it meets them, before it checks required flags
    let args = match run::prescan(&argv[1..]) {
        run::Prescan::Help => {
            let _ = run::Args::command().print_help();
            return;
        }
        run::Prescan::Version => {
            println!("{}", run::REFERENCE_VERSION);
            return;
        }
        run::Prescan::Error(e) => e.exit(),
        run::Prescan::Args(args) => args,
    };
    let args = run::Args::parse_from(std::iter::once(argv[0].clone()).chain(args));
    if let Err(e) = run::run(args) {
        log!("{e}");
        std::process::exit(1);
    }
}
