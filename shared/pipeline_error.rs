use std::error::Error;

/// Error propagated across genotype I/O and scoring pipeline stages.
#[derive(Debug, Clone)]
pub enum PipelineError {
    Compute(String),
    Io(String),
    Producer(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compute(error) => write!(f, "{error}"),
            Self::Io(error) => write!(f, "I/O error during pipeline execution: {error}"),
            Self::Producer(error) => write!(f, "The data producer thread failed: {error}"),
        }
    }
}

impl Error for PipelineError {}

impl From<Box<dyn Error + Send + Sync>> for PipelineError {
    fn from(error: Box<dyn Error + Send + Sync>) -> Self {
        Self::Compute(error.to_string())
    }
}
