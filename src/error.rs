/// Typed error variants for programmatic error handling.
///
/// Callers can match on specific variants to handle failures differently
/// (e.g., retry with `--no-coverage`, report the missing symbol, fall back).
/// The `Other` variant wraps `anyhow::Error` for gradual migration — not
/// every internal error needs a typed variant immediately.
#[derive(Debug, thiserror::Error)]
pub enum TruantError {
    #[error("unsupported binary format")]
    UnsupportedFormat,

    #[error("{}", format_corrupt(.offset, .message))]
    CorruptInput {
        offset: Option<u64>,
        message: String,
    },

    #[error("no basic blocks found to instrument")]
    NoBasicBlocks,

    #[error("instruction relocation failed at VA 0x{va:x}: {reason}")]
    RelocationFailed { va: u64, reason: String },

    #[error("hook target not found: {symbol}")]
    HookTargetNotFound { symbol: String },

    #[error("PE header space exhausted")]
    PeHeaderFull,

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

fn format_corrupt(offset: &Option<u64>, message: &str) -> String {
    match offset {
        Some(off) => format!("corrupt input at offset 0x{off:x}: {message}"),
        None => format!("corrupt input: {message}"),
    }
}

/// Convenience alias for `Result<T, TruantError>`.
pub type Result<T> = std::result::Result<T, TruantError>;
