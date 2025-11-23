"""
Pytest configuration and shared fixtures for redis_cache tests.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_deps():
    """
    Patches all external dependencies and returns a dictionary of mocks
    for detailed assertion checking.
    """
    with (
        patch("llama_index.tools.redis_cache.base.RedisVectorStore") as mock_redis,
        patch("llama_index.tools.redis_cache.base.VectorStoreIndex") as mock_index_cls,
        patch("llama_index.tools.redis_cache.base.StorageContext") as mock_storage,
        patch("llama_index.tools.redis_cache.base.IndexSchema") as mock_schema,
    ):
        # Setup the Index chain
        mock_index_instance = MagicMock()
        mock_index_cls.from_vector_store.return_value = mock_index_instance
        mock_retriever = MagicMock()
        mock_index_instance.as_retriever.return_value = mock_retriever

        # Setup Redis Store internals
        mock_vector_store = MagicMock()
        mock_redis_client = MagicMock()
        mock_vector_store.client = mock_redis_client
        mock_redis.return_value = mock_vector_store

        # Setup Schema internals (needed for add_to_cache)
        mock_schema_obj = MagicMock()
        mock_schema_obj.index.prefix = "llm_cache"
        mock_schema_obj.index.key_separator = ":"
        mock_vector_store.schema = mock_schema_obj

        yield {
            "redis_store_cls": mock_redis,
            "index_cls": mock_index_cls,
            "index_instance": mock_index_instance,
            "retriever": mock_retriever,
            "redis_client": mock_redis_client,
            "vector_store": mock_vector_store,
            "schema": mock_schema,
        }


@pytest.fixture
def tool(mock_deps):
    """Returns an initialized tool instance using the default mocks."""
    from llama_index.tools.redis_cache import RedisCacheToolSpec

    return RedisCacheToolSpec(vector_dims=1536, ttl=3600, similarity_threshold=0.9)
