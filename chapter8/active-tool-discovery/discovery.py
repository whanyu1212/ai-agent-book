"""
主动工具发现的核心：用 OpenAI 嵌入向量相似度，从 126 个工具里
按自然语言"能力需求"检索出最相关的 3-5 个候选工具。

- 工具向量：对每个工具用 "name: description" 生成 embedding，并缓存到本地
  .cache/tool_embeddings.json，避免每次运行都重新计算。
- discover_tools(need)：把 need 向量化，与工具向量做余弦相似度，返回 top-k。
"""

import hashlib
import json
import os
from typing import Dict, List, Tuple

from tools_library import ALL_TOOLS

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "tool_embeddings.json")


def _tool_text(tool: Dict) -> str:
    f = tool["function"]
    return f"{f['name']}: {f['description']}"


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


class ToolIndex:
    """工具向量索引 + 相似度检索。"""

    def __init__(self, client):
        self.client = client
        self.names = [t["function"]["name"] for t in ALL_TOOLS]
        self.texts = [_tool_text(t) for t in ALL_TOOLS]
        self.vectors = self._load_or_build()

    def _signature(self) -> str:
        h = hashlib.sha256()
        h.update(EMBED_MODEL.encode())
        for t in self.texts:
            h.update(t.encode())
        return h.hexdigest()[:16]

    def _load_or_build(self) -> Dict[str, List[float]]:
        sig = self._signature()
        if os.path.exists(_CACHE_FILE):
            try:
                cached = json.load(open(_CACHE_FILE, encoding="utf-8"))
                if cached.get("signature") == sig:
                    return cached["vectors"]
            except Exception:
                pass
        # 缓存缺失或失效 -> 调用 embedding API 批量生成
        print(f"[discovery] 正在为 {len(self.texts)} 个工具生成嵌入向量 ...")
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=self.texts)
        vectors = {name: d.embedding for name, d in zip(self.names, resp.data)}
        os.makedirs(_CACHE_DIR, exist_ok=True)
        json.dump({"signature": sig, "vectors": vectors},
                  open(_CACHE_FILE, "w", encoding="utf-8"))
        return vectors

    def search(self, need: str, top_k: int = 4) -> List[Tuple[str, float]]:
        """返回与 need 最相关的 top_k 个 (工具名, 相似度)。"""
        q = self.client.embeddings.create(model=EMBED_MODEL, input=[need]).data[0].embedding
        scored = [(name, _cosine(q, self.vectors[name])) for name in self.names]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
