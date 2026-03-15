#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // MachOContext::parse should return Err for malformed input, never panic.
    let _ = truant::macho::MachOContext::parse(data);
});
