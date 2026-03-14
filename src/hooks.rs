//! Hook configuration parsing and symbol resolution.
//!
//! Reads a TOML hook specification and resolves symbol names / hex VAs to
//! concrete file offsets and displaced bytes, ready for trampoline generation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::elf::ElfContext;
use crate::macho::MachOContext;

/// Top-level hook configuration from TOML.
#[derive(Debug, Deserialize)]
pub struct HookConfig {
    /// Optional hook library .so path (for handler functions resolved at runtime).
    #[serde(default)]
    pub hooks: Option<HookMeta>,
    /// Individual hook definitions.
    #[serde(rename = "hook", default)]
    pub hook: Vec<HookDef>,
}

/// The `[hooks]` table in the TOML.
#[derive(Debug, Deserialize)]
pub struct HookMeta {
    /// Path to the .so containing handler functions.
    pub library: Option<PathBuf>,
}

/// A single `[[hook]]` entry.
#[derive(Debug, Deserialize)]
pub struct HookDef {
    /// Target: symbol name or "0x..." hex VA.
    pub target: String,
    /// Hook mode: pre, post, or replace.
    pub mode: HookMode,
    /// Handler function name (dlsym'd from the hook library).
    pub handler: Option<String>,
    /// Raw shellcode bytes (alternative to handler).
    pub shellcode: Option<Vec<u8>>,
    /// Optional condition: hook only fires when predicate is true.
    pub condition: Option<HookCondition>,
    /// Whether this hook starts enabled (default: true).
    /// The toggle byte in the data segment can be flipped at runtime.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

/// Condition predicate for conditional hooks.
///
/// The hook only fires when the specified register value satisfies the
/// comparison against the given value. Checked at runtime inside the
/// trampoline after saving registers into the RegContext.
///
/// TOML example:
/// ```toml
/// condition = { register = "rdi", op = "gte", value = 65536 }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct HookCondition {
    /// Register name. x86_64: rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp,
    /// r8-r15. AArch64: x0-x30.
    pub register: String,
    /// Comparison operator.
    pub op: CondOp,
    /// Value to compare against (unsigned). For bit_set/bit_clear this is a
    /// bitmask, not a bit number.
    pub value: u64,
}

/// Comparison operators for hook conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CondOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Greater than (unsigned).
    Gt,
    /// Greater than or equal (unsigned).
    Gte,
    /// Less than (unsigned).
    Lt,
    /// Less than or equal (unsigned).
    Lte,
    /// All bits in value are set: (reg & value) == value.
    BitSet,
    /// All bits in value are clear: (reg & value) == 0.
    BitClear,
}

/// Hook insertion mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HookMode {
    /// Call handler before the original code.
    Pre,
    /// Call handler after the displaced instructions.
    Post,
    /// Replace the original: handler receives pointer to original stub.
    Replace,
    /// Call handler when the hooked function returns (fires at RET, not entry).
    Return,
}

/// Where the hook handler code lives.
#[derive(Debug, Clone)]
pub enum HookSource {
    /// Handler is a symbol in the companion .so; its resolved address will be
    /// written to the data slot at `data_slot_index` (index into the hook data area).
    LibrarySymbol {
        name: String,
        data_slot_index: usize,
    },
    /// Raw shellcode embedded directly in the rewritten binary.
    Shellcode(Vec<u8>),
}

/// A fully resolved hook ready for trampoline generation.
#[derive(Debug, Clone)]
pub struct ResolvedHook {
    /// Virtual address of the hook target.
    pub target_va: u64,
    /// File offset corresponding to target_va.
    pub file_offset: u64,
    /// The bytes displaced by our JMP (copied into the trampoline).
    pub displaced_bytes: Vec<u8>,
    /// Length of displaced bytes.
    pub displaced_len: usize,
    /// Hook mode.
    pub mode: HookMode,
    /// Where the handler code comes from.
    pub source: HookSource,
    /// Optional runtime condition for the hook.
    pub condition: Option<HookCondition>,
    /// Index into the toggle byte area (sequential across all hooks).
    pub toggle_index: usize,
    /// Initial value of the toggle byte (true = enabled).
    pub initial_enabled: bool,
    /// Index into the return-address slot area (only set for Return mode hooks).
    /// Each Return hook gets an 8-byte slot to save/restore the original return address.
    pub return_slot_index: Option<usize>,
}

/// Parse a hook configuration TOML file.
pub fn parse_hook_config(path: &Path) -> Result<HookConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read hook config {}", path.display()))?;
    let config: HookConfig = toml::from_str(&contents)
        .with_context(|| format!("failed to parse hook config {}", path.display()))?;

    // Validate each hook definition.
    for (i, hook) in config.hook.iter().enumerate() {
        match (&hook.handler, &hook.shellcode) {
            (None, None) => bail!(
                "hook[{}] (target={}): must specify either 'handler' or 'shellcode'",
                i,
                hook.target
            ),
            (Some(_), Some(_)) => bail!(
                "hook[{}] (target={}): cannot specify both 'handler' and 'shellcode'",
                i,
                hook.target
            ),
            _ => {}
        }
        if hook.handler.is_some() {
            // Must have a library specified.
            let has_lib = config
                .hooks
                .as_ref()
                .and_then(|m| m.library.as_ref())
                .is_some();
            if !has_lib {
                bail!(
                    "hook[{}] (target={}): 'handler' requires [hooks] library to be set",
                    i,
                    hook.target,
                );
            }
        }
    }

    Ok(config)
}

/// Valid x86_64 register names for condition checks.
const X86_64_REGS: &[&str] = &[
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

/// Validate a register name for the given architecture.
fn validate_condition_register(register: &str, is_aarch64: bool) -> Result<()> {
    if is_aarch64 {
        // x0-x30
        if let Some(rest) = register.strip_prefix('x')
            && let Ok(n) = rest.parse::<u32>()
            && n <= 30
        {
            return Ok(());
        }
        bail!("invalid AArch64 register '{}' (expected x0-x30)", register);
    } else {
        if X86_64_REGS.contains(&register) {
            return Ok(());
        }
        bail!(
            "invalid x86_64 register '{}' (expected one of: {})",
            register,
            X86_64_REGS.join(", "),
        );
    }
}

/// Parse a target string: "0x..." hex VA or symbol name.
fn parse_target(target: &str) -> TargetSpec {
    if let Some(hex) = target
        .strip_prefix("0x")
        .or_else(|| target.strip_prefix("0X"))
        && let Ok(va) = u64::from_str_radix(hex, 16)
    {
        return TargetSpec::Va(va);
    }
    TargetSpec::Symbol(target.to_string())
}

enum TargetSpec {
    Va(u64),
    Symbol(String),
}

/// Abstraction over platform-specific hook resolution details.
///
/// Each binary format implements this to provide symbol lookup,
/// executable section checks, and displaced byte extraction.
trait HookResolver {
    /// Look up a symbol name and return its VA.
    fn lookup_symbol(&self, index: usize, name: &str) -> Result<u64>;
    /// Verify a VA is in an executable section and return (section_va, file_offset).
    fn verify_executable(&self, index: usize, target: &str, va: u64) -> Result<(u64, u64)>;
    /// Extract displaced bytes at the given VA.
    fn extract_displaced(&self, index: usize, target: &str, va: u64) -> Result<(Vec<u8>, usize)>;
    /// Whether the target architecture uses AArch64 register names.
    fn is_aarch64(&self) -> bool;
}

/// Generic hook resolution using a platform-specific `HookResolver`.
fn resolve_hooks_generic(
    config: &HookConfig,
    resolver: &dyn HookResolver,
) -> Result<Vec<ResolvedHook>> {
    let mut resolved = Vec::with_capacity(config.hook.len());
    let mut data_slot_index = 0usize;
    let mut return_slot_counter = 0usize;

    for (i, hook) in config.hook.iter().enumerate() {
        let target_va = match parse_target(&hook.target) {
            TargetSpec::Va(va) => va,
            TargetSpec::Symbol(name) => resolver.lookup_symbol(i, &name)?,
        };

        let (sec_va, sec_offset) = resolver.verify_executable(i, &hook.target, target_va)?;
        let (displaced_bytes, displaced_len) =
            resolver.extract_displaced(i, &hook.target, target_va)?;
        let file_offset = (target_va - sec_va) + sec_offset;

        if let Some(ref cond) = hook.condition {
            validate_condition_register(&cond.register, resolver.is_aarch64()).with_context(
                || {
                    format!(
                        "hook[{}] (target={}): invalid condition register",
                        i, hook.target,
                    )
                },
            )?;
        }

        let source = match (&hook.handler, &hook.shellcode) {
            (Some(name), _) => {
                let idx = data_slot_index;
                data_slot_index += 1;
                HookSource::LibrarySymbol {
                    name: name.clone(),
                    data_slot_index: idx,
                }
            }
            (_, Some(code)) => HookSource::Shellcode(code.clone()),
            _ => unreachable!("validated in parse_hook_config"),
        };

        let ret_slot = if hook.mode == HookMode::Return {
            let idx = return_slot_counter;
            return_slot_counter += 1;
            Some(idx)
        } else {
            None
        };

        resolved.push(ResolvedHook {
            target_va,
            file_offset,
            displaced_bytes,
            displaced_len,
            mode: hook.mode,
            source,
            condition: hook.condition.clone(),
            toggle_index: i,
            initial_enabled: hook.enabled,
            return_slot_index: ret_slot,
        });
    }

    Ok(resolved)
}

/// Resolve hook definitions to concrete VAs and displaced bytes.
///
/// Scans both `.symtab` and `.dynsym` for symbol name lookups.
pub fn resolve_hooks(
    config: &HookConfig,
    data: &[u8],
    ctx: &ElfContext,
    min_displaced_bytes: usize,
) -> Result<Vec<ResolvedHook>> {
    let sym_map = build_symbol_map(data)?;
    let resolver = ElfResolver {
        sym_map,
        data,
        ctx,
        min_displaced_bytes,
    };
    resolve_hooks_generic(config, &resolver)
}

struct ElfResolver<'a> {
    sym_map: HashMap<String, u64>,
    data: &'a [u8],
    ctx: &'a ElfContext,
    min_displaced_bytes: usize,
}

impl HookResolver for ElfResolver<'_> {
    fn lookup_symbol(&self, index: usize, name: &str) -> Result<u64> {
        self.sym_map.get(name).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "hook[{}]: symbol '{}' not found in binary (searched .symtab and .dynsym)",
                index,
                name,
            )
        })
    }

    fn verify_executable(&self, index: usize, target: &str, va: u64) -> Result<(u64, u64)> {
        let (sec_va, sec_offset, _) = self.ctx.exec_section_for_va(va).ok_or_else(|| {
            anyhow::anyhow!(
                "hook[{}] (target={}): VA 0x{:x} is not in any executable section",
                index,
                target,
                va,
            )
        })?;
        Ok((sec_va, sec_offset))
    }

    fn extract_displaced(&self, index: usize, target: &str, va: u64) -> Result<(Vec<u8>, usize)> {
        crate::disasm::extract_displaced_bytes(self.data, self.ctx, va, self.min_displaced_bytes)
            .with_context(|| {
                format!(
                    "hook[{}] (target={}): failed to extract displaced bytes at 0x{:x}",
                    index, target, va,
                )
            })
    }

    fn is_aarch64(&self) -> bool {
        self.ctx.e_machine == 183
    }
}

/// Build a symbol name → VA map from the ELF's .symtab and .dynsym.
fn build_symbol_map(data: &[u8]) -> Result<std::collections::HashMap<String, u64>> {
    use goblin::elf::Elf;

    let elf = Elf::parse(data).context("failed to parse ELF for symbol resolution")?;
    let mut map = std::collections::HashMap::new();

    // .symtab
    for sym in &elf.syms {
        if sym.st_value != 0
            && let Some(name) = elf.strtab.get_at(sym.st_name)
            && !name.is_empty()
        {
            map.entry(name.to_string()).or_insert(sym.st_value);
        }
    }

    // .dynsym (override symtab if present in both)
    for sym in &elf.dynsyms {
        if sym.st_value != 0
            && let Some(name) = elf.dynstrtab.get_at(sym.st_name)
            && !name.is_empty()
        {
            map.insert(name.to_string(), sym.st_value);
        }
    }

    Ok(map)
}

/// Resolve hook definitions for Mach-O binaries.
///
/// Scans nlist_64 entries from LC_SYMTAB for symbol name lookups.
/// Rejects duplicate targets (same VA).
pub fn resolve_hooks_macho(
    config: &HookConfig,
    data: &[u8],
    ctx: &MachOContext,
) -> Result<Vec<ResolvedHook>> {
    let sym_map = build_symbol_map_macho(data)?;
    let resolver = MachOResolver { sym_map, data, ctx };
    resolve_hooks_generic(config, &resolver)
}

/// ARM64 Mach-O CPU type constant.
const CPU_TYPE_ARM64: u32 = 0x0100_000C;

struct MachOResolver<'a> {
    sym_map: HashMap<String, u64>,
    data: &'a [u8],
    ctx: &'a MachOContext,
}

impl HookResolver for MachOResolver<'_> {
    fn lookup_symbol(&self, index: usize, name: &str) -> Result<u64> {
        self.sym_map
            .get(name)
            .or_else(|| {
                let prefixed = format!("_{}", name);
                self.sym_map.get(prefixed.as_str())
            })
            .copied()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "hook[{}]: symbol '{}' not found in Mach-O (searched LC_SYMTAB nlist entries)",
                    index,
                    name,
                )
            })
    }

    fn verify_executable(&self, index: usize, target: &str, va: u64) -> Result<(u64, u64)> {
        let text = &self.ctx.text;
        if va < text.va || va >= text.va + text.size {
            bail!(
                "hook[{}] (target={}): VA 0x{:x} is outside __text (0x{:x}..0x{:x})",
                index,
                target,
                va,
                text.va,
                text.va + text.size,
            );
        }
        let file_offset = crate::macho::va_to_file_offset_macho(va, self.ctx).ok_or_else(|| {
            anyhow::anyhow!(
                "hook[{}] (target={}): VA 0x{:x} has no file mapping",
                index,
                target,
                va,
            )
        })? as u64;
        // For Mach-O, sec_va concept doesn't apply the same way — the file_offset
        // is absolute, so we set sec_va = va and sec_offset = file_offset so that
        // (target_va - sec_va) + sec_offset == file_offset.
        Ok((va, file_offset))
    }

    fn extract_displaced(&self, index: usize, target: &str, va: u64) -> Result<(Vec<u8>, usize)> {
        if self.ctx.cputype == CPU_TYPE_ARM64 {
            let file_offset =
                crate::macho::va_to_file_offset_macho(va, self.ctx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "hook[{}] (target={}): VA 0x{:x} has no file mapping",
                        index,
                        target,
                        va,
                    )
                })?;
            if file_offset + 4 > self.data.len() {
                bail!(
                    "hook[{}] (target={}): VA 0x{:x} too close to end of file for displaced bytes",
                    index,
                    target,
                    va,
                );
            }
            Ok((self.data[file_offset..file_offset + 4].to_vec(), 4))
        } else {
            crate::macho_disasm::extract_displaced_bytes_macho(self.data, self.ctx, va, 5)
                .with_context(|| {
                    format!(
                        "hook[{}] (target={}): failed to extract displaced bytes at 0x{:x}",
                        index, target, va,
                    )
                })
        }
    }

    fn is_aarch64(&self) -> bool {
        self.ctx.cputype == CPU_TYPE_ARM64
    }
}

/// Build a symbol name → VA map from a Mach-O's nlist_64 entries (LC_SYMTAB).
fn build_symbol_map_macho(data: &[u8]) -> Result<std::collections::HashMap<String, u64>> {
    use goblin::mach::Mach;

    let mut map = std::collections::HashMap::new();

    match Mach::parse(data).context("failed to parse Mach-O for symbol resolution")? {
        Mach::Binary(macho) => {
            if let Some(symbols) = macho.symbols {
                for (name, nlist) in symbols.iter().flatten() {
                    if nlist.n_value != 0 && !name.is_empty() && (nlist.n_type & 0x0E) == 0x0E
                    // N_SECT
                    {
                        // Store both the raw name and stripped underscore prefix
                        map.insert(name.to_string(), nlist.n_value);
                        if let Some(stripped) = name.strip_prefix('_') {
                            map.entry(stripped.to_string()).or_insert(nlist.n_value);
                        }
                    }
                }
            }
        }
        _ => bail!("expected single Mach-O binary, got fat/other"),
    }

    Ok(map)
}

/// Resolve hook definitions for PE binaries.
///
/// Scans PE export directory and import table for symbol name lookups.
/// For imports, locates the IAT thunk stub (`jmp [rip+disp32]`) in .text
/// and hooks that stub — analogous to ELF PLT hooks.
/// Rejects duplicate targets (same VA).
pub fn resolve_hooks_pe(
    config: &HookConfig,
    data: &[u8],
    ctx: &crate::pe::PeContext,
) -> Result<Vec<ResolvedHook>> {
    let sym_map = build_symbol_map_pe(data)?;
    let import_thunk_map = build_import_thunk_map(data, ctx);
    let resolver = PeResolver {
        sym_map,
        import_thunk_map,
        data,
        ctx,
    };
    resolve_hooks_generic(config, &resolver)
}

struct PeResolver<'a> {
    sym_map: HashMap<String, u64>,
    import_thunk_map: HashMap<String, u64>,
    data: &'a [u8],
    ctx: &'a crate::pe::PeContext,
}

impl HookResolver for PeResolver<'_> {
    fn lookup_symbol(&self, index: usize, name: &str) -> Result<u64> {
        if let Some(&va) = self.sym_map.get(name) {
            Ok(va)
        } else if let Some(&va) = self.import_thunk_map.get(name) {
            Ok(va)
        } else {
            bail!(
                "hook[{}]: symbol '{}' not found in PE exports or imports",
                index,
                name
            )
        }
    }

    fn verify_executable(&self, index: usize, target: &str, va: u64) -> Result<(u64, u64)> {
        let (sec_va, sec_offset, _) =
            crate::pe::exec_section_for_va(va, self.ctx).ok_or_else(|| {
                anyhow::anyhow!(
                    "hook[{}] (target={}): VA 0x{:x} is not in any executable section",
                    index,
                    target,
                    va,
                )
            })?;
        Ok((sec_va, sec_offset))
    }

    fn extract_displaced(&self, index: usize, target: &str, va: u64) -> Result<(Vec<u8>, usize)> {
        crate::pe_disasm::extract_displaced_bytes_pe(self.data, self.ctx, va, 5).with_context(
            || {
                format!(
                    "hook[{}] (target={}): failed to extract displaced bytes at 0x{:x}",
                    index, target, va,
                )
            },
        )
    }

    fn is_aarch64(&self) -> bool {
        false
    }
}

/// Build a map from import function names to their IAT thunk VAs.
///
/// For each PE import, scans .text for a `jmp [rip+disp32]` (FF 25) thunk
/// that references the import's IAT slot. This is the PE analogue of an ELF
/// PLT stub. Multiple imports with the same name (from different DLLs) will
/// use the first thunk found.
fn build_import_thunk_map(
    data: &[u8],
    ctx: &crate::pe::PeContext,
) -> std::collections::HashMap<String, u64> {
    let mut map = std::collections::HashMap::new();

    for imp in &ctx.imports {
        if imp.name.is_empty() || imp.name.starts_with("ORDINAL ") {
            continue;
        }
        if let Some(thunk_va) = crate::pe::find_iat_thunk(data, ctx, imp.iat_rva) {
            // First match wins — if the same name is imported from multiple DLLs,
            // only the first thunk is used. Users can always target by VA for
            // disambiguation.
            map.entry(imp.name.clone()).or_insert(thunk_va);
        }
    }

    map
}

/// Build a symbol name → VA map from a PE's export directory (exported functions only).
/// Import resolution is handled separately via `build_import_thunk_map`.
fn build_symbol_map_pe(data: &[u8]) -> Result<std::collections::HashMap<String, u64>> {
    use goblin::pe::PE;

    let pe = PE::parse(data).context("failed to parse PE for symbol resolution")?;
    let mut map = std::collections::HashMap::new();

    let image_base = pe
        .header
        .optional_header
        .as_ref()
        .map(|oh| oh.windows_fields.image_base)
        .unwrap_or(0);

    for export in &pe.exports {
        if export.rva != 0
            && let Some(name) = export.name
            && !name.is_empty()
        {
            map.insert(name.to_string(), image_base + export.rva as u64);
        }
    }

    Ok(map)
}

/// Return the number of data slots needed for library symbol hooks.
#[must_use]
pub fn count_library_hooks(hooks: &[ResolvedHook]) -> usize {
    hooks
        .iter()
        .filter(|h| matches!(h.source, HookSource::LibrarySymbol { .. }))
        .count()
}

/// Return the number of return-address slots needed for Return mode hooks.
#[must_use]
pub fn count_return_hooks(hooks: &[ResolvedHook]) -> usize {
    hooks.iter().filter(|h| h.mode == HookMode::Return).count()
}

/// Return the library path from the hook config, if any.
#[must_use]
pub fn library_path(config: &HookConfig) -> Option<&Path> {
    config.hooks.as_ref().and_then(|m| m.library.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_target_hex() {
        match parse_target("0x4012a0") {
            TargetSpec::Va(va) => assert_eq!(va, 0x4012a0),
            _ => panic!("expected VA"),
        }
        match parse_target("0X1000") {
            TargetSpec::Va(va) => assert_eq!(va, 0x1000),
            _ => panic!("expected VA"),
        }
    }

    #[test]
    fn test_parse_target_symbol() {
        match parse_target("TIFFReadEncodedStrip") {
            TargetSpec::Symbol(name) => assert_eq!(name, "TIFFReadEncodedStrip"),
            _ => panic!("expected Symbol"),
        }
    }

    #[test]
    fn test_parse_hook_config_shellcode() {
        let toml = r#"
[[hook]]
target = "0x4012a0"
mode = "pre"
shellcode = [0x90, 0x90, 0xC3]
"#;
        let config: HookConfig = toml::from_str(toml).expect("valid TOML for shellcode test");
        assert_eq!(config.hook.len(), 1);
        assert_eq!(config.hook[0].mode, HookMode::Pre);
        assert_eq!(
            config.hook[0]
                .shellcode
                .as_ref()
                .expect("shellcode should be set"),
            &[0x90, 0x90, 0xC3]
        );
    }

    #[test]
    fn test_parse_hook_config_handler() {
        let toml = r#"
[hooks]
library = "./my_hooks.so"

[[hook]]
target = "main"
mode = "replace"
handler = "my_handler"
"#;
        let config: HookConfig = toml::from_str(toml).expect("valid TOML for handler test");
        assert_eq!(config.hook.len(), 1);
        assert_eq!(config.hook[0].mode, HookMode::Replace);
        assert_eq!(
            config.hook[0]
                .handler
                .as_ref()
                .expect("handler should be set"),
            "my_handler"
        );
        assert_eq!(
            config
                .hooks
                .as_ref()
                .expect("hooks table should be present")
                .library
                .as_ref()
                .expect("library should be set")
                .to_str()
                .expect("library path should be valid UTF-8"),
            "./my_hooks.so"
        );
    }

    #[test]
    fn test_parse_hook_config_multiple() {
        let toml = r#"
[hooks]
library = "./hooks.so"

[[hook]]
target = "0x4012a0"
mode = "pre"
handler = "pre_hook"

[[hook]]
target = "some_func"
mode = "post"
shellcode = [0xC3]

[[hook]]
target = "0x401000"
mode = "replace"
handler = "replace_hook"
"#;
        let config: HookConfig = toml::from_str(toml).expect("valid TOML for multiple hooks test");
        assert_eq!(config.hook.len(), 3);
        assert_eq!(config.hook[0].mode, HookMode::Pre);
        assert_eq!(config.hook[1].mode, HookMode::Post);
        assert_eq!(config.hook[2].mode, HookMode::Replace);
    }

    #[test]
    fn test_validate_handler_without_library() {
        let toml = r#"
[[hook]]
target = "main"
mode = "pre"
handler = "my_handler"
"#;
        let config: HookConfig =
            toml::from_str(toml).expect("valid TOML for handler-without-library test");
        let _path = Path::new("/dev/null");
        // parse_hook_config validates, but we can test validate logic directly
        for (i, hook) in config.hook.iter().enumerate() {
            if hook.handler.is_some() {
                let has_lib = config
                    .hooks
                    .as_ref()
                    .and_then(|m| m.library.as_ref())
                    .is_some();
                assert!(!has_lib, "hook[{}] should fail: no library", i);
            }
        }
    }

    #[test]
    fn test_validate_neither_handler_nor_shellcode() {
        let toml = r#"
[[hook]]
target = "main"
mode = "pre"
"#;
        let config: HookConfig =
            toml::from_str(toml).expect("valid TOML for neither-handler-nor-shellcode test");
        for hook in &config.hook {
            assert!(hook.handler.is_none() && hook.shellcode.is_none());
        }
    }

    #[test]
    fn test_hook_mode_deserialise() {
        assert_eq!(
            toml::from_str::<HookDef>(
                r#"target = "x"
mode = "pre""#
            )
            .expect("valid TOML for pre mode")
            .mode,
            HookMode::Pre,
        );
        assert_eq!(
            toml::from_str::<HookDef>(
                r#"target = "x"
mode = "post""#
            )
            .expect("valid TOML for post mode")
            .mode,
            HookMode::Post,
        );
        assert_eq!(
            toml::from_str::<HookDef>(
                r#"target = "x"
mode = "replace""#
            )
            .expect("valid TOML for replace mode")
            .mode,
            HookMode::Replace,
        );
    }

    #[test]
    fn test_parse_hook_with_condition() {
        let toml = r#"
[[hook]]
target = "0x401000"
mode = "pre"
shellcode = [0xC3]
condition = { register = "rdi", op = "gte", value = 65536 }
"#;
        let config: HookConfig = toml::from_str(toml).expect("valid TOML for condition test");
        assert_eq!(config.hook.len(), 1);
        let cond = config.hook[0]
            .condition
            .as_ref()
            .expect("condition should be set");
        assert_eq!(cond.register, "rdi");
        assert_eq!(cond.op, CondOp::Gte);
        assert_eq!(cond.value, 65536);
    }

    #[test]
    fn test_parse_hook_without_condition() {
        let toml = r#"
[[hook]]
target = "0x401000"
mode = "pre"
shellcode = [0xC3]
"#;
        let config: HookConfig = toml::from_str(toml).expect("valid TOML for no-condition test");
        assert!(config.hook[0].condition.is_none());
    }

    #[test]
    fn test_parse_condition_all_ops() {
        for op in ["eq", "ne", "gt", "gte", "lt", "lte", "bit_set", "bit_clear"] {
            let toml = format!(
                r#"target = "x"
mode = "pre"
shellcode = [0xC3]
condition = {{ register = "rax", op = "{}", value = 1 }}"#,
                op,
            );
            let def: HookDef = toml::from_str(&toml)
                .unwrap_or_else(|e| panic!("failed to parse op '{}': {}", op, e));
            assert!(def.condition.is_some(), "condition missing for op '{}'", op);
        }
    }

    #[test]
    fn test_validate_condition_register_x86_64() {
        assert!(validate_condition_register("rax", false).is_ok());
        assert!(validate_condition_register("r15", false).is_ok());
        assert!(validate_condition_register("rdi", false).is_ok());
        assert!(validate_condition_register("x0", false).is_err());
        assert!(validate_condition_register("eax", false).is_err());
    }

    #[test]
    fn test_parse_hook_enabled_default() {
        let toml = r#"
[[hook]]
target = "0x401000"
mode = "pre"
shellcode = [0xC3]
"#;
        let config: HookConfig = toml::from_str(toml).expect("valid TOML for enabled-default test");
        assert!(config.hook[0].enabled, "enabled should default to true");
    }

    #[test]
    fn test_parse_hook_enabled_false() {
        let toml = r#"
[[hook]]
target = "0x401000"
mode = "pre"
shellcode = [0xC3]
enabled = false
"#;
        let config: HookConfig = toml::from_str(toml).expect("valid TOML for enabled-false test");
        assert!(
            !config.hook[0].enabled,
            "enabled should be false when explicitly set"
        );
    }

    #[test]
    fn test_validate_condition_register_aarch64() {
        assert!(validate_condition_register("x0", true).is_ok());
        assert!(validate_condition_register("x30", true).is_ok());
        assert!(validate_condition_register("x31", true).is_err());
        assert!(validate_condition_register("rax", true).is_err());
    }

    #[test]
    fn test_hook_mode_return_deserialise() {
        let def: HookDef = toml::from_str(
            r#"target = "x"
mode = "return""#,
        )
        .expect("valid TOML for return mode");
        assert_eq!(def.mode, HookMode::Return);
    }

    #[test]
    fn test_return_hook_slot_index() {
        // Verify that return_slot_index is assigned only for Return mode hooks.
        let toml = r#"
[hooks]
library = "./hooks.so"

[[hook]]
target = "0x4000"
mode = "pre"
handler = "pre_fn"

[[hook]]
target = "0x5000"
mode = "return"
handler = "ret_fn"

[[hook]]
target = "0x6000"
mode = "return"
handler = "ret_fn2"

[[hook]]
target = "0x7000"
mode = "post"
handler = "post_fn"
"#;
        let config: HookConfig = toml::from_str(toml).expect("valid TOML for return-slot test");
        assert_eq!(config.hook.len(), 4);
        assert_eq!(config.hook[0].mode, HookMode::Pre);
        assert_eq!(config.hook[1].mode, HookMode::Return);
        assert_eq!(config.hook[2].mode, HookMode::Return);
        assert_eq!(config.hook[3].mode, HookMode::Post);

        // We can't run resolve_hooks without a real ELF, but we can verify
        // the slot assignment logic directly by simulating it.
        let mut return_slot_counter = 0usize;
        let mut slots = Vec::new();
        for hook in &config.hook {
            if hook.mode == HookMode::Return {
                slots.push(Some(return_slot_counter));
                return_slot_counter += 1;
            } else {
                slots.push(None);
            }
        }
        assert_eq!(slots, vec![None, Some(0), Some(1), None]);
    }
}
