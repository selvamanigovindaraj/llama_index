import pytest
from unittest.mock import MagicMock, AsyncMock
from llama_index.tools.redis_cache import RedisCacheToolSpec
from llama_index.core.schema import TextNode


# Test constants - should match the values set in conftest.py mock_deps fixture
EXPECTED_KEY_PREFIX = "llm_cache"
EXPECTED_KEY_SEPARATOR = ":"


# --- Initialization Tests ---


def test_init_with_vector_dims(mock_deps):
    """Test initialization with vector_dims parameter."""
    tool = RedisCacheToolSpec(vector_dims=1536, index_name="test_cache", ttl=7200)

    # Verify dependencies were called with correct arguments
    mock_deps["schema"].from_dict.assert_called_once()
    schema_call = mock_deps["schema"].from_dict.call_args[0][0]
    assert schema_call["index"]["name"] == "test_cache"

    mock_deps["redis_store_cls"].assert_called_once()
    mock_deps["index_cls"].from_vector_store.assert_called_once()

    # Verify behavior: tool should be able to perform its operations
    # (existence of spec_functions indicates proper initialization)
    assert hasattr(tool, "check_cache")
    assert hasattr(tool, "add_to_cache")
    assert callable(tool.check_cache)
    assert callable(tool.add_to_cache)


def test_init_with_vector_store(mock_deps):
    """Test initialization with existing vector_store."""
    mock_vector_store = MagicMock()
    mock_index_instance = MagicMock()
    mock_deps["index_cls"].from_vector_store.return_value = mock_index_instance
    mock_retriever = MagicMock()
    mock_index_instance.as_retriever.return_value = mock_retriever

    tool = RedisCacheToolSpec(vector_store=mock_vector_store)

    # Should not create a new vector store
    assert mock_deps["redis_store_cls"].call_count == 0

    # Verify the provided vector_store was used to create the index
    mock_deps["index_cls"].from_vector_store.assert_called_once()
    call_args = mock_deps["index_cls"].from_vector_store.call_args
    assert call_args[0][0] is mock_vector_store  # First positional arg

    # Verify tool has required functionality
    assert callable(tool.check_cache)
    assert callable(tool.add_to_cache)


def test_init_without_vector_dims_or_store():
    """Test initialization fails without vector_dims or vector_store."""
    with pytest.raises(ValueError, match="vector_dims must be provided"):
        RedisCacheToolSpec()


def test_init_with_custom_params(mock_deps):
    """Test initialization with custom parameters."""
    mock_embed_model = MagicMock()

    tool = RedisCacheToolSpec(
        vector_dims=3072,
        embed_model=mock_embed_model,
        index_name="custom_cache",
        prefix="custom_prefix",
        redis_url="redis://custom:6379",
        ttl=7200,
        similarity_threshold=0.95,
    )

    # Verify schema was created (high-level check)
    mock_deps["schema"].from_dict.assert_called_once()
    schema_call = mock_deps["schema"].from_dict.call_args[0][0]

    # Verify high-level parameters were passed correctly
    assert schema_call["index"]["name"] == "custom_cache"
    assert schema_call["index"]["prefix"] == "custom_prefix"

    # Verify vector dimensions were configured (without assuming field order)
    assert any(
        f.get("type") == "vector" and f.get("attrs", {}).get("dims") == 3072
        for f in schema_call.get("fields", [])
    ), "Vector field with dims=3072 not found in schema"


# --- Sync Cache Operation Tests ---


def test_check_cache_hit(mock_deps, tool):
    """Test check_cache returns the metadata response on high score."""
    mock_node = MagicMock(score=0.95)
    mock_node.metadata = {"response": "Cached response"}
    mock_deps["retriever"].retrieve.return_value = [mock_node]

    result = tool.check_cache("test query")

    assert result == "Cached response"
    mock_deps["retriever"].retrieve.assert_called_once_with("test query")


@pytest.mark.parametrize(
    ("threshold", "node_score", "expected_result", "description"),
    [
        (0.9, 0.91, "Cached response", "Score above threshold -> Hit"),
        (0.9, 0.90, "Cached response", "Score equals threshold -> Hit"),
        (0.9, 0.89, None, "Score below threshold -> Miss"),
        (0.9, None, None, "Score is None -> Miss"),
        (0.5, 0.49, None, "Lower threshold, score below -> Miss"),
        (0.5, 0.50, "Cached response", "Lower threshold, score equals -> Hit"),
    ],
)
def test_check_cache_boundary_logic(
    mock_deps, threshold, node_score, expected_result, description
):
    """
    Test cache hit/miss logic with explicit thresholds to ensure
    boundary conditions are respected regardless of fixture defaults.
    """
    # Initialize a tool with a specific threshold for this test case
    tool = RedisCacheToolSpec(vector_dims=1536, similarity_threshold=threshold)

    # Setup the mock return
    mock_node = MagicMock()
    mock_node.score = node_score
    mock_node.metadata = {"response": "Cached response"}
    mock_deps["retriever"].retrieve.return_value = [mock_node]

    # Act
    result = tool.check_cache("test query")

    # Assert
    assert result == expected_result, f"Failed on case: {description}"
    mock_deps["retriever"].retrieve.assert_called_once_with("test query")


def test_check_cache_empty_results(mock_deps, tool):
    """Test check_cache with no results from retriever."""
    mock_deps["retriever"].retrieve.return_value = []

    result = tool.check_cache("test query")

    assert result is None


def test_add_to_cache(mock_deps, tool):
    """Test adding to cache verifies correct node construction and TTL expiration."""
    node_id = tool.add_to_cache("test query", "test response")

    # Assert: Check correct Node construction
    mock_deps["index_instance"].insert_nodes.assert_called_once()
    inserted_node = mock_deps["index_instance"].insert_nodes.call_args[0][0][0]
    assert isinstance(inserted_node, TextNode)
    assert inserted_node.text == "test query"
    assert inserted_node.metadata["response"] == "test response"
    assert "response" in inserted_node.excluded_embed_metadata_keys
    assert "response" in inserted_node.excluded_llm_metadata_keys

    # Assert: Check TTL usage with correct arguments
    mock_deps["redis_client"].expire.assert_called_once()
    expire_call_args = mock_deps["redis_client"].expire.call_args

    # Verify the key contains expected components (without reconstructing implementation)
    actual_key = expire_call_args[0][0]
    assert EXPECTED_KEY_PREFIX in actual_key, (
        f"Key should contain prefix '{EXPECTED_KEY_PREFIX}'"
    )
    assert inserted_node.node_id in actual_key, "Key should contain node_id"
    assert EXPECTED_KEY_SEPARATOR in actual_key, (
        f"Key should contain separator '{EXPECTED_KEY_SEPARATOR}'"
    )

    # Verify the TTL value
    assert expire_call_args[0][1] == 3600

    # Assert: Return value is the node_id
    assert node_id == inserted_node.node_id


def test_add_to_cache_custom_ttl(mock_deps):
    """Test add_to_cache uses custom TTL value."""
    tool = RedisCacheToolSpec(vector_dims=1536, ttl=7200)

    tool.add_to_cache("query", "response")

    # Verify custom TTL is used
    expire_call_args = mock_deps["redis_client"].expire.call_args
    assert expire_call_args[0][1] == 7200


# --- Async Cache Operation Tests ---


@pytest.mark.asyncio
async def test_acheck_cache_hit(mock_deps, tool):
    """Test async check_cache returns cached response."""
    mock_node = MagicMock(score=0.95)
    mock_node.metadata = {"response": "Async cached response"}
    mock_deps["retriever"].aretrieve = AsyncMock(return_value=[mock_node])

    result = await tool.acheck_cache("test query")

    assert result == "Async cached response"
    mock_deps["retriever"].aretrieve.assert_called_once_with("test query")


@pytest.mark.asyncio
async def test_acheck_cache_miss(mock_deps, tool):
    """Test async check_cache with no results."""
    mock_deps["retriever"].aretrieve = AsyncMock(return_value=[])

    result = await tool.acheck_cache("test query")

    assert result is None


@pytest.mark.asyncio
async def test_aadd_to_cache(mock_deps, tool):
    """Test async add_to_cache functionality."""
    node_id = await tool.aadd_to_cache("async query", "async response")

    # Verify node insertion
    mock_deps["index_instance"].insert_nodes.assert_called_once()
    inserted_node = mock_deps["index_instance"].insert_nodes.call_args[0][0][0]
    assert inserted_node.text == "async query"
    assert inserted_node.metadata["response"] == "async response"

    # Verify expire was called with correct TTL
    mock_deps["redis_client"].expire.assert_called_once()
    expire_call_args = mock_deps["redis_client"].expire.call_args
    assert expire_call_args[0][1] == 3600  # Verify TTL value

    assert node_id is not None


# --- Spec Functions Test ---


def test_spec_functions(tool):
    """Test that spec_functions are correctly defined."""
    assert "check_cache" in tool.spec_functions
    assert "add_to_cache" in tool.spec_functions
    assert "acheck_cache" in tool.spec_functions
    assert "aadd_to_cache" in tool.spec_functions
    assert len(tool.spec_functions) == 4
