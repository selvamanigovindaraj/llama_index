from unittest.mock import MagicMock, patch
from llama_index.tools.redis_cache import RedisCacheToolSpec


@patch("llama_index.tools.redis_cache.base.RedisVectorStore")
@patch("llama_index.tools.redis_cache.base.VectorStoreIndex")
@patch("llama_index.tools.redis_cache.base.StorageContext")
def test_redis_cache_tool_spec(mock_storage_context, mock_index_cls, mock_redis_store):
    # Setup mocks
    mock_index_instance = MagicMock()
    mock_index_cls.from_vector_store.return_value = mock_index_instance

    mock_retriever = MagicMock()
    mock_index_instance.as_retriever.return_value = mock_retriever

    # Initialize tool
    tool = RedisCacheToolSpec(index_name="cache_index")

    # Verify initialization
    mock_redis_store.assert_called_once()
    mock_index_cls.from_vector_store.assert_called_once()

    # Test check_cache - Hit
    mock_node = MagicMock()
    mock_node.score = 0.95
    mock_node.metadata = {"response": "Cached response"}
    mock_retriever.retrieve.return_value = [mock_node]

    result = tool.check_cache("test query")
    assert result == "Cached response"

    # Test check_cache - Miss (Low score)
    mock_node.score = 0.5
    result = tool.check_cache("test query")
    assert result is None

    # Test check_cache - Miss (No results)
    mock_retriever.retrieve.return_value = []
    result = tool.check_cache("test query")
    assert result is None

    # Test add_to_cache
    mock_redis_client = MagicMock()
    mock_redis_store.return_value.client = mock_redis_client
    # Mock internal index attributes for key construction
    mock_redis_store.return_value._index.prefix = "cache"
    mock_redis_store.return_value._index.key_separator = ":"

    tool.add_to_cache("new query", "new response")

    mock_index_instance.insert_nodes.assert_called_once()
    mock_redis_client.expire.assert_called_once()
