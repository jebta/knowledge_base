import json
from minio import Minio

from config.config import MinioConfig

minio_client = None


def get_minio_client():
    global minio_client
    if not minio_client:
        minio_client = Minio(
            endpoint=MinioConfig.MINIO_ENDPOINT,
            access_key=MinioConfig.MINIO_ACCESS_KEY,
            secret_key=MinioConfig.MINIO_SECRET_KEY,
            secure=False
        )
    minio_bucket_name = MinioConfig.MINIO_BUCKET_NAME
    if not minio_client.bucket_exists(bucket_name=minio_bucket_name):
        minio_client.make_bucket(bucket_name=minio_bucket_name)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                "Resource": f"arn:aws:s3:::{minio_bucket_name}",
            },
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{minio_bucket_name}/*",
            },
        ],
    }
    minio_client.set_bucket_policy(bucket_name=minio_bucket_name, policy=json.dumps(policy))

    return minio_client
