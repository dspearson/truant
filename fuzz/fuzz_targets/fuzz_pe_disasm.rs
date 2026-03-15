#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Parse then disassemble — the full PE pipeline should never panic.
    if let Ok(ctx) = truant::pe::PeContext::parse(data) {
        let _ = truant::pe_disasm::find_basic_blocks_pe(&ctx, data, &None);
    }
});
