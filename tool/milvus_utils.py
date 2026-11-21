from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from config.config import MilvusConfig

milvus_client = None


def get_milvus_client():
    global milvus_client
    if milvus_client is None:
        milvus_client = MilvusClient(uri=MilvusConfig.milvus_url)
    return milvus_client


def create_hybrid_request(dense_data, sparse_data, dense_anns_field, sparse_anns_field, dense_param=None,
                          sparse_param=None, limit=10, expr=None):
    if not dense_param:
        dense_param = {"metric_type": "COSINE"}
    if not sparse_param:
        sparse_param = {"metric_type": "IP"}
    dense_req = AnnSearchRequest(
        data=dense_data,
        anns_field=dense_anns_field,
        param=dense_param,
        limit=limit,
        expr=expr
    )
    sparse_req = AnnSearchRequest(
        data=sparse_data,
        anns_field=sparse_anns_field,
        param=sparse_param,
        limit=limit,
        expr=expr
    )
    return [dense_req, sparse_req]


def hybrid_search(collection_name, reqs, weight=(0.5, 0.5), limit=10, output_fields=None):
    milvus_client = get_milvus_client()
    ranker = WeightedRanker(*weight)
    res = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=limit,
        output_fields=output_fields,
    )
    return res
