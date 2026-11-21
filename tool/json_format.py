import json

from bson import ObjectId


def json_format(data):
    class MongoJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, ObjectId):
                return str(o)  # 将 ObjectId 转为 24位十六进制字符串
            return super().default(o)

    return json.dumps(data, indent=4, ensure_ascii=False, cls=MongoJSONEncoder)
