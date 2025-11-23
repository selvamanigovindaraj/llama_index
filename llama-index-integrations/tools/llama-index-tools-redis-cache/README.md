# LlamaIndex Tools Integration: Redis Cache

This tool provides Redis Semantic Search capabilities for LlamaIndex agents.

## Installation

```bash
pip install llama-index-tools-redis-cache
```

## Usage

```python
from llama_index.tools.redis_cache import RedisCacheToolSpec
from llama_index.agent.openai import OpenAIAgent

# Initialize the tool
tool_spec = RedisCacheToolSpec(
    index_name="cache",
    prefix="cache",
    redis_url="redis://localhost:6379",
    ttl=3600,
    similarity_threshold=0.9,
)

# Create an agent with the cache tool
agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())

# Use the agent
response = agent.chat("What is machine learning?")
```
