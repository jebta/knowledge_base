from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from config.config import BGEM3Config

bge_m3_model = None


def get_bge_m3_model():
    global bge_m3_model
    if not bge_m3_model:
        bge_m3_model = BGEM3EmbeddingFunction(
            model_name=BGEM3Config.bge_m3_path,
            device=BGEM3Config.bge_device,
            use_fp16=True if BGEM3Config.bge_fp16 in ("True", "1", 1, True) else False
        )
    return bge_m3_model


def get_bge_m3_embeddings(texts):
    bge_m3_model = get_bge_m3_model()
    result = bge_m3_model.encode_documents(texts)
    dense = result.get("dense")
    sparse = result.get("sparse")
    return {
        "dense": [item.tolist() for item in dense],
        "sparse": [dict(zip(item.indices.tolist(), item.data.tolist())) for item in sparse]
    }


if __name__ == '__main__':
    print(get_bge_m3_embeddings(["你好", "世界"]))
