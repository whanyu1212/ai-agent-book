"""Configuration for Agentic RAG User Memory Evaluation System"""

import os
from dataclasses import dataclass, field, replace
from typing import ClassVar, Optional
from enum import Enum
from dotenv import load_dotenv

from agentbook.providers import (
    PROVIDERS,
    Backend,
    canonical_provider,
    map_model_to_openrouter,
)
from agentbook.providers import resolve_backend as resolve_provider_backend

load_dotenv()


def _reasoning_safe_temperature(model, requested=1.0):
    """Reasoning models (Kimi K3, GPT-5, ...) only accept temperature=1.
    Return 1 for those; otherwise the requested value so non-reasoning
    providers (Doubao, DeepSeek, older Moonshot) are unchanged."""
    m = str(model or "").lower().replace("/", "-")
    return 1 if ("kimi-k3" in m or "gpt-5" in m) else requested


class Provider(str, Enum):
    """Supported LLM providers"""
    SILICONFLOW = "siliconflow"
    DOUBAO = "doubao"
    KIMI = "kimi"
    MOONSHOT = "moonshot"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GROQ = "groq"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"


class IndexMode(str, Enum):
    """Indexing modes for conversation chunks"""
    DENSE = "dense"  # Dense embedding only
    SPARSE = "sparse"  # Sparse embedding only (BM25)
    HYBRID = "hybrid"  # Both dense and sparse


class ChunkingStrategy(str, Enum):
    """Strategies for chunking conversations"""
    FIXED_ROUNDS = "fixed_rounds"  # Fixed number of rounds per chunk
    SEMANTIC = "semantic"  # Semantic boundaries
    TIME_BASED = "time_based"  # Based on timestamp gaps


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "kimi"  # Default provider
    model: Optional[str] = None  # Will use provider defaults if not specified
    api_key: Optional[str] = None  # Will read from env if not provided
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = True
    
    # Experiment-specific defaults stay local; credentials, endpoints, aliases,
    # and fallback behavior are owned by agentbook.providers.
    PROVIDER_MODEL_DEFAULTS: ClassVar[dict[str, str]] = {
        "siliconflow": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "doubao": "doubao-seed-1-6-thinking-250715",
        "kimi": "kimi-k3",
        "openrouter": "openai/gpt-5.6-luna",
        "openai": "gpt-5.6-luna",
        "groq": "llama-3.3-70b-versatile",
        "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "deepseek": "deepseek-reasoner",
    }

    def resolve_backend(self) -> Backend:
        """Resolve the configured provider into a ready-to-use backend."""
        provider = canonical_provider(self.provider)
        model = self.model or self.PROVIDER_MODEL_DEFAULTS.get(provider)
        provider_spec = PROVIDERS.get(provider)
        legacy_doubao_key = os.getenv("DOUBAO_API_KEY", "").strip()
        ark_key = os.getenv("ARK_API_KEY", "").strip()
        api_key = (self.api_key or "").strip() or (
            (ark_key or legacy_doubao_key) if provider == "doubao" else None
        )
        primary_key = api_key or (provider_spec.api_key() if provider_spec else "")
        backend = resolve_provider_backend(provider, model=model, api_key=api_key)

        # Keep direct-provider credentials ahead of the resolver's GPT-5 route.
        if primary_key and provider != "openrouter" and backend.using_openrouter:
            direct_model = model or backend.model
            if provider_spec.namespaces_models:
                direct_model = map_model_to_openrouter(direct_model)
            backend = Backend(primary_key, provider_spec.resolved_base_url(), direct_model,
                              provider_spec.name, False)

        override = os.getenv("OPENROUTER_MODEL", "").strip()
        using_fallback = backend.using_openrouter and not primary_key and provider != "openrouter"
        if using_fallback and override:
            return replace(backend, model=override)
        if (not using_fallback and provider_spec and provider_spec.namespaces_models and model
                and map_model_to_openrouter(model) == model and backend.model != model):
            return replace(backend, model=model)
        return backend

    def get_client_config(self) -> tuple[dict[str, str], str]:
        """Return the legacy OpenAI-client tuple from the resolved backend."""
        backend = self.resolve_backend()
        return {"api_key": backend.api_key, "base_url": backend.base_url}, backend.model


@dataclass
class ChunkingConfig:
    """Configuration for conversation chunking"""
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_ROUNDS
    rounds_per_chunk: int = 20  # Number of rounds per chunk for FIXED_ROUNDS
    overlap_rounds: int = 2  # Number of overlapping rounds between chunks
    include_metadata: bool = True  # Include conversation metadata in chunks
    min_chunk_size: int = 5  # Minimum number of rounds in a chunk
    max_chunk_size: int = 50  # Maximum number of rounds in a chunk


@dataclass
class IndexConfig:
    """Configuration for RAG indexing"""
    mode: IndexMode = IndexMode.HYBRID
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model
    embedding_dim: int = 1536  # Dimension of embeddings
    index_path: str = "indexes/memory_index"
    chunk_store_path: str = "data/chunk_store.json"
    enable_contextual: bool = True  # Add contextual information to chunks
    contextual_window: int = 2  # Number of surrounding rounds for context


@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework"""
    test_cases_dir: str = "../../week2/user-memory-evaluation/test_cases"
    results_dir: str = "results"
    enable_verbose: bool = True
    save_trajectories: bool = True
    max_iterations: int = 10  # Max iterations for ReAct pattern
    enable_caching: bool = True  # Cache indexed conversations
    use_llm_judge: bool = False  # Use LLM to evaluate answers


@dataclass
class AgentConfig:
    """Agent behavior configuration"""
    enable_reasoning: bool = True  # Show reasoning steps
    enable_citations: bool = True  # Include citations in responses
    max_search_results: int = 5  # Maximum search results to consider
    confidence_threshold: float = 0.7  # Minimum confidence for answers
    enable_multi_search: bool = True  # Allow multiple searches per query
    max_searches_per_query: int = 3  # Maximum searches allowed
    verbose: bool = True  # Enable verbose output


@dataclass
class Config:
    """Main configuration container"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        if provider := os.getenv("LLM_PROVIDER"):
            config.llm.provider = provider
        
        if model := os.getenv("LLM_MODEL"):
            config.llm.model = model
        
        if rounds := os.getenv("ROUNDS_PER_CHUNK"):
            config.chunking.rounds_per_chunk = int(rounds)
        
        if index_mode := os.getenv("INDEX_MODE"):
            config.index.mode = IndexMode(index_mode)
        
        return config
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        import json
        
        config_dict = {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": _reasoning_safe_temperature(self.llm.model, self.llm.temperature),
                "max_tokens": self.llm.max_tokens,
                "stream": self.llm.stream
            },
            "chunking": {
                "strategy": self.chunking.strategy,
                "rounds_per_chunk": self.chunking.rounds_per_chunk,
                "overlap_rounds": self.chunking.overlap_rounds,
                "include_metadata": self.chunking.include_metadata
            },
            "index": {
                "mode": self.index.mode,
                "embedding_model": self.index.embedding_model,
                "enable_contextual": self.index.enable_contextual,
                "contextual_window": self.index.contextual_window
            },
            "evaluation": {
                "enable_verbose": self.evaluation.enable_verbose,
                "save_trajectories": self.evaluation.save_trajectories,
                "max_iterations": self.evaluation.max_iterations
            },
            "agent": {
                "enable_reasoning": self.agent.enable_reasoning,
                "enable_citations": self.agent.enable_citations,
                "max_search_results": self.agent.max_search_results,
                "confidence_threshold": self.agent.confidence_threshold
            }
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from JSON file"""
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Update LLM config
        if "llm" in config_dict:
            for key, value in config_dict["llm"].items():
                setattr(config.llm, key, value)
        
        # Update other configs similarly
        for section in ["chunking", "index", "evaluation", "agent"]:
            if section in config_dict:
                section_config = getattr(config, section)
                for key, value in config_dict[section].items():
                    # Handle enums
                    if key == "strategy" and section == "chunking":
                        value = ChunkingStrategy(value)
                    elif key == "mode" and section == "index":
                        value = IndexMode(value)
                    setattr(section_config, key, value)
        
        return config
