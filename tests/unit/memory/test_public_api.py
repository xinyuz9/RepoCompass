"""Public memory package surface tests."""

import repocompass.memory as memory


def test_memory_package_exports_stable_m5_public_boundaries() -> None:
    expected = {
        "EpisodicMemory",
        "EpisodeConsolidationResult",
        "MemoryRepositoryIdentity",
        "MemoryRetrievalHit",
        "MemoryStore",
        "RepositoryMemory",
        "RepositoryMemorySearchResult",
        "classify_repository_memory_freshness",
        "consolidate_terminal_episode",
        "create_episodic_memory",
        "create_repository_memory",
        "open_memory_store",
        "resolve_memory_repository_identity",
    }

    assert expected.issubset(memory.__all__)
    assert all(hasattr(memory, name) for name in memory.__all__)
    assert memory.MEMORY_CONTRACT_SCHEMA_VERSION == 1
    assert memory.MEMORY_STORE_SCHEMA_VERSION == 2
    assert memory.EPISODE_CONSOLIDATION_SCHEMA_VERSION == 1
