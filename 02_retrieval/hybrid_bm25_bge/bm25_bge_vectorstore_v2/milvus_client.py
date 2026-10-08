# 模块名称：milvus_client.py —— 本地 Milvus-Lite 客户端（兼容现有接口）
from typing import Dict, List, Optional, Iterable, Set, Any
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import warnings
import os


class MilvusClient:
    def __init__(
        self,
        uri: str,
        user: str = "",
        password: str = "",
        db_name: str = "default",
        index_dense_cfg: Optional[Dict] = None,
        index_sparse_cfg: Optional[Dict] = None,
        alias: str = "default",
    ):
        """
        统一用 uri 连接：
          - 若 uri 是本地路径（例如 /xx/books.db），则以 Milvus-Lite 方式连接；
          - 若是 http://host:port 也可兼容，但本项目推荐本地 .db。
        """
        self.uri = uri
        self.alias = alias
        self.db_name = db_name
        self.index_dense_cfg = index_dense_cfg or {
            "type": "HNSW",
            "metric": "IP",
            "params": {"M": 32, "efConstruction": 200},
            "search_params": {"ef": 256},
        }
        self.index_sparse_cfg = index_sparse_cfg or {
            "type": "SPARSE_INVERTED_INDEX",
            "metric": "IP",
            "params": {},
        }

        # 连接（始终优先 uri）
        try:
            connections.connect(alias=self.alias, uri=self.uri, user=user, password=password, db_name=db_name)
        except Exception as e:
            raise RuntimeError(f"Failed to connect Milvus via uri={self.uri}: {e}")

    # ---------- collection ops ----------

    def _build_schema(
        self,
        dims_dense: int,
        max_text_len: int,
        max_summary_len: int,
    ) -> CollectionSchema:
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=8),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=max_text_len),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=max_summary_len),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dims_dense),
        ]
        return CollectionSchema(fields, description="BM25 + BGE hybrid store (Lite)")

    def _is_lite(self) -> bool:
        # 经验规则：本地 Lite 通常传文件路径，或以 "file://" 开头
        u = (self.uri or "").lower()
        return not u.startswith("http://") and not u.startswith("https://") and not u.startswith("tcp://")

    def _create_indexes(self, col: Collection):
        # 稀疏索引保持尝试（Lite 目前支持 SPARSE_INVERTED_INDEX；若不支持会被 warn 捕捉）
        try:
            col.create_index(
                field_name="sparse_vector",
                index_params={
                    "index_type": self.index_sparse_cfg.get("type", "SPARSE_INVERTED_INDEX"),
                    "metric_type": self.index_sparse_cfg.get("metric", "IP"),
                    "params": self.index_sparse_cfg.get("params", {}),
                },
            )
        except Exception as e:
            warnings.warn(f"[milvus_client] create sparse index failed or exists: {e}")

        # 稠密索引：Lite 下不支持 HNSW，强制降级为 IVF_FLAT（或 AUTOINDEX）
        dense_type = (self.index_dense_cfg.get("type") or "IVF_FLAT").upper()
        dense_metric = self.index_dense_cfg.get("metric", "IP")
        dense_params = self.index_dense_cfg.get("params", {})
        if self._is_lite() and dense_type not in ("FLAT", "IVF_FLAT", "AUTOINDEX"):
            warnings.warn(f"[milvus_client] Lite mode: force dense index to IVF_FLAT (was {dense_type})")
            dense_type = "IVF_FLAT"
            dense_params = dense_params or {"nlist": 1024}

        try:
            col.create_index(
                field_name="dense_vector",
                index_params={
                    "index_type": dense_type,
                    "metric_type": dense_metric,
                    "params": dense_params,
                },
            )
        except Exception as e:
            warnings.warn(f"[milvus_client] create dense index failed or exists: {e}")

    def ensure_partitions(self, col: Collection, partitions: Iterable[str]):
        if self._is_lite():
            warnings.warn("[milvus_client] Lite mode: partitions are not supported; skip creating partitions.")
            return
        for p in partitions:
            try:
                if not utility.has_partition(collection_name=col.name, partition_name=p):
                    col.create_partition(p)
            except Exception as e:
                warnings.warn(f"[milvus_client] create partition {p} warn: {e}")


    def ensure_collection(
        self,
        collection: str,
        dims_dense: int,
        max_text_len: int,
        max_summary_len: int,
        partitions: Iterable[str],
    ) -> Collection:
        """
        若集合不存在则创建（含索引与分区）；若已存在则校验维度。
        """
        if utility.has_collection(collection):
            col = Collection(collection)
            # 校验维度
            dim_field = [f for f in col.schema.fields if f.name == "dense_vector"]
            if not dim_field:
                raise ValueError(f"Collection {collection} exists but without dense_vector field")
            exist_dim = dim_field[0].params.get("dim", None)
            if exist_dim and int(exist_dim) != int(dims_dense):
                raise ValueError(f"Collection {collection} dim={exist_dim} != expected {dims_dense}")
            # 确保索引与分区
            self._create_indexes(col)
            self.ensure_partitions(col, partitions)
            try:
                col.load()
            except Exception:
                pass
            return col

        # 创建集合
        schema = self._build_schema(dims_dense, max_text_len, max_summary_len)
        col = Collection(name=collection, schema=schema)
        self._create_indexes(col)
        self.ensure_partitions(col, partitions)
        try:
            col.load()
        except Exception:
            pass
        return col

    def get_collection(self, collection: str) -> Collection:
        if not utility.has_collection(collection):
            raise ValueError(f"Collection {collection} not found")
        col = Collection(collection)
        try:
            col.load()
        except Exception:
            pass
        return col

    # ---------- insert / query ops ----------

    def insert_rows(self, col: Collection, rows: List[Dict[str, Any]], partition: Optional[str] = None):
        """
        将任意 {"indices": [...], "values": [...]} 或已是 {idx: val} 的稀疏向量
        统一规整为 {int(idx): float(val)} 再插入。
        """
        def _to_sparse_map(s):
            # {"indices":[...], "values":[...]} -> {idx: val}
            if isinstance(s, dict) and "indices" in s and "values" in s:
                return {int(i): float(v) for i, v in zip(s["indices"], s["values"]) if v}
            # 已是 {idx: val} 或 None
            if isinstance(s, dict):
                return {int(k): float(v) for k, v in s.items() if v}
            return {}

        if not rows:
            return

        # 逐行规整：尤其是 sparse_vector & dense_vector
        normed: List[Dict[str, Any]] = []
        for r in rows:
            r2 = dict(r)
            r2["sparse_vector"] = _to_sparse_map(r2.get("sparse_vector", {}))

            # 稠密向量确保为 List[float]
            dv = r2.get("dense_vector", [])
            if dv is None:
                dv = []
            if not isinstance(dv, list):
                try:
                    dv = list(dv)
                except Exception:
                    dv = []
            r2["dense_vector"] = [float(x) for x in dv]

            normed.append(r2)

        # 带分区（云端）优先；Lite 无分区
        if partition and not self._is_lite():
            try:
                col.insert(normed, partition_name=partition)
                return
            except Exception as e:
                warnings.warn(f"[milvus_client] insert with partition {partition} failed, fallback without partition: {e}")

        # Lite 或分区失败：无分区插入
        col.insert(normed)


    def query_existing_by_hash(self, col: Collection, hashes: List[str]) -> Set[str]:
        """
        返回已经存在的 text_hash 集合，用于去重
        """
        if not hashes:
            return set()
        # Milvus 支持 in 表达式，但注意分批避免过长
        existed: Set[str] = set()
        step = 1000
        for i in range(0, len(hashes), step):
            chunk = hashes[i : i + step]
            expr = f'text_hash in {chunk}'
            try:
                res = col.query(expr=expr, output_fields=["text_hash"])
                existed.update([r["text_hash"] for r in res])
            except Exception:
                # 某些 Lite 版本对 in 的字符串处理严格，这里回退逐个查询（较慢，只在异常时触发）
                for h in chunk:
                    try:
                        r = col.query(expr=f'text_hash == "{h}"', output_fields=["text_hash"])
                        if r:
                            existed.add(h)
                    except Exception:
                        pass
        return existed

    # ---------- search ops ----------

    def search_dense(self, col: Collection, query_vecs: List[List[float]], topk: int, search_dense_cfg: Dict):
        metric = search_dense_cfg.get("metric", search_dense_cfg.get("metric_type", "IP"))
        params = search_dense_cfg.get("search_params", search_dense_cfg.get("params", {"ef": 256}))
        light = os.getenv("MILVUS_LIGHT_OUTPUT", "0") == "1"
        fields = ["pk"] if light else ["pk", "text", "summary", "file_name", "metadata", "lang"]

        return col.search(
            data=query_vecs,
            anns_field="dense_vector",
            param={"metric_type": metric, "params": params},
            limit=topk,
            output_fields=fields,
        )


    # milvus_client.py

    # bm25_bge_vectorstore_v2/milvus_client.py

    def search_sparse(self, col: Collection, q_sparse_list: List[Dict], topk: int):
        """
        q_sparse_list: List[{"indices":[...], "values":[...]}]  或  List[{idx: val, ...}]
        """
        # ✅ 统一成 {idx: val}，Milvus-Lite 稳定支持这种格式
        def _to_sparse_map(s):
            if isinstance(s, dict) and "indices" in s and "values" in s:
                return {int(i): float(v) for i, v in zip(s["indices"], s["values"]) if v}
            return s or {}

        q_mapped = [_to_sparse_map(s) for s in q_sparse_list]

        light = os.getenv("MILVUS_LIGHT_OUTPUT", "0") == "1"
        fields = ["pk"] if light else ["pk", "text", "summary", "file_name", "metadata", "lang"]
        
        return col.search(
            data=q_mapped,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=topk,
            output_fields=fields,
        )
        



