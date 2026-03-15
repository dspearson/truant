#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // PeContext::parse should return Err for malformed input, never panic.
    let _ = truant::pe::PeContext::parse(data);
});
