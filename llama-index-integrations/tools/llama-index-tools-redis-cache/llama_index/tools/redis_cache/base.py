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

        self._index.insert_nodes([node])

        client = self._vector_store.client
        schema = self._vector_store.schema
        prefix = schema.index.prefix
        key_separator = schema.index.key_separator

        key = f"{prefix}{key_separator}{node.node_id}"

        import redis.asyncio as redis_async

        if isinstance(client, redis_async.Redis):
            await client.expire(key, self._ttl)
        else:
            client.expire(key, self._ttl)

        return node.node_id
