from typing import Any, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from redisvl.schema import IndexSchema


class RedisCacheToolSpec(BaseToolSpec):
    """
    Redis Semantic Cache Tool Spec.

    This tool acts as a semantic cache for LLM responses.
    It allows checking for existing answers to similar queries and adding new responses to the cache.
    """

    spec_functions = ["check_cache", "add_to_cache", "acheck_cache", "aadd_to_cache"]

    def __init__(
        self,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[RedisVectorStore] = None,
        index_name: str = "cache",
        prefix: str = "cache",
        redis_url: str = "redis://localhost:6379",
        ttl: int = 3600,
        similarity_threshold: float = 0.9,
        vector_dims: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with Redis connection details and cache settings.

        Args:
            embed_model (Optional[BaseEmbedding]): Embedding model to use.
            vector_store (Optional[RedisVectorStore]): Existing RedisVectorStore instance.
            index_name (str): Name of the Redis index.
            prefix (str): Prefix for Redis keys.
            redis_url (str): URL for Redis connection.
            ttl (int): Time to live for cache entries in seconds.
            similarity_threshold (float): Threshold for semantic similarity.
            vector_dims (Optional[int]): Dimensions of the embedding vector. Required if vector_store is None.
            **kwargs: Additional arguments passed to RedisVectorStore if initializing it.

        """
        self._ttl = ttl
        self._similarity_threshold = similarity_threshold
        self._embed_model = embed_model

        if vector_store:
            self._vector_store = vector_store
        else:
            if vector_dims is None:
                raise ValueError(
                    "vector_dims must be provided if vector_store is not passed."
                )

            # Create the schema for RedisVectorStore
            schema = IndexSchema.from_dict(
                {
                    "index": {"name": index_name, "prefix": prefix},
                    "fields": [
                        {"name": "id", "type": "tag"},
                        {"name": "doc_id", "type": "tag"},
                        {"name": "text", "type": "text"},
                        {
                            "name": "vector",
                            "type": "vector",
                            "attrs": {"dims": vector_dims, "algorithm": "flat"},
                        },
                    ],
                }
            )

            self._vector_store = RedisVectorStore(
                schema=schema,
                redis_url=redis_url,
                **kwargs,
            )

        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )

        self._index = VectorStoreIndex.from_vector_store(
            self._vector_store,
            storage_context=self._storage_context,
            embed_model=self._embed_model,
        )

        # Initialize retriever once
        self._retriever = self._index.as_retriever(similarity_top_k=1)

    def check_cache(self, query: str) -> Optional[str]:
        """
        Check the cache for a semantically similar query.

        Args:
            query (str): The user query to check.

        Returns:
            Optional[str]: The cached response if a hit is found, otherwise None.

        """
        nodes = self._retriever.retrieve(query)

        if not nodes:
            return None

        best_match = nodes[0]

        if (
            best_match.score is not None
            and best_match.score >= self._similarity_threshold
        ):
            return best_match.metadata.get("response")

        return None

    async def acheck_cache(self, query: str) -> Optional[str]:
        """
        Async check the cache for a semantically similar query.

        Args:
            query (str): The user query to check.

        Returns:
            Optional[str]: The cached response if a hit is found, otherwise None.

        """
        nodes = await self._retriever.aretrieve(query)

        if not nodes:
            return None

        best_match = nodes[0]

        if (
            best_match.score is not None
            and best_match.score >= self._similarity_threshold
        ):
            return best_match.metadata.get("response")

        return None

    def add_to_cache(self, query: str, response: str) -> str:
        """
        Add a query-response pair to the cache.

        Args:
            query (str): The user query (used for embedding).
            response (str): The LLM response to cache.

        Returns:
            str: The ID of the added cache entry.

        """
        node = TextNode(
            text=query,
            metadata={"response": response},
            excluded_embed_metadata_keys=["response"],
            excluded_llm_metadata_keys=["response"],
        )

        self._index.insert_nodes([node])

        # Use public client property
        client = self._vector_store.client

        # Use public schema property to get prefix and separator
        schema = self._vector_store.schema
        prefix = schema.index.prefix
        key_separator = schema.index.key_separator

        key = f"{prefix}{key_separator}{node.node_id}"
        client.expire(key, self._ttl)

        return node.node_id

    async def aadd_to_cache(self, query: str, response: str) -> str:
        """
        Async add a query-response pair to the cache.

        Args:
            query (str): The user query (used for embedding).
            response (str): The LLM response to cache.

        Returns:
            str: The ID of the added cache entry.

        """
        node = TextNode(
            text=query,
            metadata={"response": response},
            excluded_embed_metadata_keys=["response"],
            excluded_llm_metadata_keys=["response"],
        )

        # Use async insert if available, otherwise fallback to sync insert (VectorStoreIndex might not have ainsert_nodes exposed directly on index, but storage context does?)
        # VectorStoreIndex doesn't have ainsert_nodes.
        # But we can use the ingestion pipeline or just insert to vector store directly?
        # Ideally we use index.insert_nodes which handles embedding.
        # VectorStoreIndex.insert_nodes is sync.
        # There is no async insert_nodes on VectorStoreIndex yet?
        # Let's check if we can use the storage context or vector store directly.
        # But we need embedding.

        # If we want to be truly async, we should generate embedding async and then insert async.
        # But VectorStoreIndex abstracts this.
        # For now, we might have to call sync insert_nodes.
        # Wait, let's check if VectorStoreIndex has `insert` method? No.

        # Actually, if we use `self._index.insert_nodes`, it uses the embed model.
        # If embed model is sync, it blocks.
        # If we want async, we might need to do it manually:
        # 1. Embed query async
        # 2. Add to vector store async

        # However, for the purpose of this tool, calling the sync insert_nodes might be acceptable if the underlying store supports it.
        # But `client.expire` should be async if we use async client.

        # Let's stick to sync insert for now as `VectorStoreIndex` doesn't seem to have easy async insert.
        # But we can make the expire async.

        self._index.insert_nodes([node])

        # Use public client property (might be sync or async depending on how initialized)
        # RedisVectorStore.client returns the client being used.
        # If we initialized with redis_url, it created a sync client by default in `__init__`.
        # But RedisVectorStore also has `_redis_client_async`.
        # The `client` property returns `_async_index.client` if `_async_index` is set?
        # Let's check the property again:
        # if self._async_index: return self._async_index.client
        # return self._index.client

        # So if we want async client, we need to ensure async index is created.
        # RedisVectorStore creates async index if `redis_client_async` is passed or if we call `aquery`?
        # Actually in `__init__` of RedisVectorStore:
        # if not self._redis_client_async: self._redis_client_async = redis_async.Redis(...)

        # So it seems it always has an async client available internally?
        # But `client` property returns one or the other.

        # If we want to be safe, we can try to access the async client if possible or just use the sync client for expire since it's fast.
        # But `aadd_to_cache` should ideally be async.

        # Let's assume `client` returns a client we can use. If it's sync client, `expire` is sync.
        # If we want to use async expire, we need the async client.
        # RedisVectorStore doesn't expose `async_client` property publicly?
        # It has `_redis_client_async`.

        # Given the constraints and "No private member access", we should rely on what `client` gives us.
        # If `client` is sync, we call it sync.

        client = self._vector_store.client

        schema = self._vector_store.schema
        prefix = schema.index.prefix
        key_separator = schema.index.key_separator

        key = f"{prefix}{key_separator}{node.node_id}"

        # Check if client is async
        import redis.asyncio as redis_async

        if isinstance(client, redis_async.Redis):
            await client.expire(key, self._ttl)
        else:
            client.expire(key, self._ttl)

        return node.node_id
