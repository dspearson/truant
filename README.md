# Truant

Static binary rewriter for arbitrary function hooking across ELF, Mach-O, and PE.

Truant rewrites compiled binaries directly — no source code, no recompilation, no dynamic instrumentation frameworks. Point it at a binary, tell it where to hook, and it produces a new binary with your hooks baked in.

## What it does

- **Hooks functions** in compiled binaries (pre-call, post-call, replace, return)
- **Supports three binary formats**: ELF (Linux), Mach-O (macOS), PE/COFF (Windows) — both 64-bit and 32-bit PE
- **No source required**: works on stripped binaries, system libraries, third-party software
- **Conditional hooks**: fire only when a register matches a predicate (e.g. `rdi >= 65536`)
- **Hook chaining**: multiple hooks on the same function, called in declaration order
- **Runtime toggle**: each hook has a 1-byte enable/disable flag in the data segment
- **Coverage instrumentation**: AFL-compatible shared memory bitmaps, forkserver, persistent mode (for fuzzing)
- **Heap sanitiser**: guard-page allocator catches overflow, use-after-free, double-free via SIGSEGV
- **Sidecar memory sanitiser**: ring-buffer logging of allocator and memory operations

## Installation

```sh
cargo install --path .
```

Or build from source:

```sh
cargo build --release
```

Requires Rust edition 2024 (stable toolchain 1.85+).

### Optional features

| Feature | Default | Description |
|---------|---------|-------------|
| `aarch64` | Yes | ARM64 support (ELF + Mach-O). Adds `capstone` dependency. |
| `coverage` | Yes | AFL-compatible coverage, forkserver, persistent mode, heap/sidecar sanitisers. |

Build without coverage (hooks-only, smaller binary):

```sh
cargo build --release --no-default-features
```

## Quick start

### Inspect a binary

```sh
truant /bin/ls --info
```

### Rewrite with coverage instrumentation

```sh
truant /bin/ls -o ls_instrumented
./ls_instrumented /tmp   # runs normally, coverage data written to SHM if AFL is driving it
```

### Hook a function by symbol name

Create a hook specification (`hooks.toml`):

```toml
[[hook]]
target = "malloc"
mode = "pre"
shellcode = [0xC3]  # handler does nothing (returns immediately)
```

Apply it:

```sh
truant /usr/bin/target -o target_hooked --hooks hooks.toml --no-coverage
./target_hooked
```

### Hook by address

```sh
# Find the address
nm /usr/bin/target | grep ' T interesting_function'
# 0000000000401234 T interesting_function

cat > hooks.toml << 'EOF'
[[hook]]
target = "0x401234"
mode = "replace"
shellcode = [0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, 0xC3]  # mov rax, 42; ret
EOF

truant /usr/bin/target -o target_hooked --hooks hooks.toml --no-coverage
```

### Hook with a shared library handler

```toml
[hooks]
library = "./my_hooks.so"

[[hook]]
target = "process_input"
mode = "pre"
handler = "my_pre_handler"

[[hook]]
target = "process_input"
mode = "post"
handler = "my_post_handler"
```

Your handler receives a pointer to the saved register context:

```c
// my_hooks.c — compile with: gcc -shared -fPIC -o my_hooks.so my_hooks.c
#include <stdio.h>
#include <stdint.h>

// x86_64 RegContext layout (144 bytes):
//   +0 rax, +8 rbx, +16 rcx, +24 rdx, +32 rsi, +40 rdi,
//   +48 rbp, +56 rsp, +64 r8, +72 r9, +80 r10, +88 r11,
//   +96 r12, +104 r13, +112 r14, +120 r15, +128 rip, +136 rflags

void my_pre_handler(uint64_t *regs) {
    printf("process_input called with rdi=%lx rsi=%lx\n", regs[5], regs[4]);
}

void my_post_handler(uint64_t *regs) {
    printf("process_input returned rax=%lx\n", regs[0]);
}
```

## Hook modes

| Mode | Description |
|------|-------------|
| `pre` | Call handler before the original function. Handler receives `&RegContext`. |
| `post` | Call handler after the displaced instructions execute. |
| `replace` | Replace the function entirely. Handler receives `(&RegContext, original_func_ptr)`. |
| `return` | Call handler when the function returns (intercepts RET). |

## Hook specification format

```toml
# Optional: shared library containing handler functions
[hooks]
library = "./my_hooks.so"

# Hook definitions
[[hook]]
target = "function_name"       # Symbol name or "0x..." hex VA
mode = "pre"                   # pre, post, replace, or return
handler = "handler_func"       # Function name in the hook library
# OR
shellcode = [0x90, 0xC3]       # Raw bytes (mutually exclusive with handler)

# Optional fields:
enabled = true                 # Initial toggle state (default: true)
condition = { register = "rdi", op = "gte", value = 65536 }
```

### Condition operators

| Operator | Description |
|----------|-------------|
| `eq`, `ne` | Equal, not equal |
| `gt`, `gte`, `lt`, `lte` | Unsigned comparison |
| `bit_set` | `(reg & value) == value` |
| `bit_clear` | `(reg & value) == 0` |

### Hook chaining

Multiple hooks on the same target are called in declaration order within a single save/restore frame:

```toml
[[hook]]
target = "0x401000"
mode = "pre"
shellcode = [0xC3]   # called first

[[hook]]
target = "0x401000"
mode = "pre"
shellcode = [0xC3]   # called second
```

Mixed pre + post chaining is supported. Replace mode is exclusive (cannot be mixed with pre/post).

## Coverage instrumentation

Truant can add AFL-compatible coverage instrumentation for fuzzing:

```sh
# Coverage rewrite (includes forkserver by default)
truant target_binary -o target_instrumented

# Persistent mode (faster — reuses process across inputs)
truant target_binary -o target_instrumented \
  --persistent-addr 0x401234 \
  --persistent-count 10000

# Deferred forkserver (fork after init, inherits loaded libraries)
truant target_binary -o target_instrumented \
  --persistent-addr 0x401234
  # (deferred by default when persistent-addr is set)
```

### Heap sanitiser

Catches heap buffer overflow, use-after-free, and double-free:

```sh
# ELF: generates a companion .so loaded via LD_PRELOAD
truant target -o target_san --heap-san
LD_PRELOAD=./target_san.heap_san.so ./target_san

# PE: companion DLL injected via IAT (loaded automatically)
truant target.exe -o target_san.exe --heap-san
# target_san.exe.heap_san.dll is generated alongside
```

## Platform support

| Feature | ELF (Linux) | Mach-O (macOS) | PE64 (Windows) | PE32 (Windows) |
|---------|:-----------:|:--------------:|:--------------:|:--------------:|
| Pre/post/replace hooks | x | x | x | x |
| Return hooks | x | x | x | x |
| Conditional hooks | x | x | x | x |
| Hook chaining | x | x | x | x |
| Runtime toggle | x | x | x | x |
| Library handler hooks | x | x | x | x |
| Coverage instrumentation | x | x | x | x |
| Persistent mode | x | x | x | x |
| Heap sanitiser | x | x | x | - |
| Sidecar sanitiser | x | x | x | - |
| Forkserver | x | x | - | - |
| AArch64 | x | x | - | - |

### ELF binary types tested

- Dynamically-linked PIE
- Dynamically-linked non-PIE
- Statically-linked (glibc)
- Statically-linked (musl)
- Shared objects (.so)

## Architecture

```
src/
  lib.rs                  Orchestration: detect format, dispatch to patchers
  patcher.rs              Shared PatchResult type
  hooks.rs                TOML config parsing, symbol resolution (generic + per-format)
  hook_trampoline.rs      Hook trampoline codegen (TargetAbi dispatch)
  trampoline.rs           Coverage trampoline + init code (ELF x86_64)
  detect.rs               Binary format and architecture detection

  elf.rs                  ELF parsing (sections, symbols, PLT, allocators)
  elf_impl/
    patcher.rs            ELF patching (new segment via PT_NOTE or phdr append, block patching, heap san)
    context.rs            ElfBinaryContext wrapper

  macho.rs                Mach-O parsing (segments, symbols, stubs)
  macho_impl/
    patcher.rs            Mach-O patching (__TR_COV/__TR_DAT segments, LINKEDIT)
    context.rs            MachOBinaryContext wrapper
  macho_disasm.rs         Mach-O basic block detection
  macho_trampoline.rs     Mach-O init code + persistent mode (x86_64 + AArch64)
  fat.rs                  Universal (fat) binary support

  pe.rs                   PE/COFF parsing (sections, imports, exports, IAT injection)
  pe_impl/
    patcher.rs            PE patching (.trcov section, PE32/PE64 init code, persistent)
    context.rs            PeBinaryContext wrapper
  pe_disasm.rs            PE basic block detection (32-bit + 64-bit)

  preload.rs              Heap sanitiser (LD_PRELOAD / DYLD_INSERT / companion DLL)
  sidecar_preload.rs      Sidecar memory sanitiser (SHM ring buffer)
  hook_preload.rs         Companion library for library-based hooks

  arch/
    x86_64/               x86_64 disassembly + trampoline generation
    aarch64/              AArch64 disassembly + trampoline generation + hooks

  traits/                 Patcher, Disassembler, TrampolineGenerator traits
  main.rs                 CLI (clap)
```

## Building and testing

```sh
cargo build              # debug build
cargo test               # 341 tests across 6 test suites (339 pass, 2 require Wine)
cargo test -- --test-threads=1   # if parallel tests flake (subprocess races)
cargo build --release    # optimised build
```

### Cross-compiling PE test binaries

PE integration tests require MinGW cross-compilers:

```sh
# 64-bit PE tests
sudo apt install gcc-mingw-w64-x86-64

# 32-bit PE tests
sudo apt install gcc-mingw-w64-i686
```

Tests skip gracefully if cross-compilers are not available.

## When should I use this?

Probably never. I wrote this as part of another project and thought it might be useful on its own, so I ripped it out into a standalone library. There are almost certainly better tools out there that can do this already — [Frida](https://frida.re/), [DynamoRIO](https://dynamorio.org/), [Pin](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-dynamic-binary-instrumentation-tool.html), probably others. Those are mature, well-tested, and maintained by people who actually understand this stuff. For me this was a learning exercise to implement what I thought would be a nice addition to something I was working on, being ignorant enough in the areas in which I was operating to not realise I was reinventing the wheel.

That said, truant does have one thing going for it: it's *static*. No runtime overhead, no agent process, no ptrace. The hooks are in the binary. The binary runs normally. If that matters to you for some reason, maybe give it a go.

## Licence

[ISC](LICENCE)
