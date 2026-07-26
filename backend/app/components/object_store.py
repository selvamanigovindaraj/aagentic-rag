from __future__ import annotations

import asyncio
from pathlib import Path

import boto3
from botocore.config import Config

from ..core.config import Settings


class S3ObjectStore:
    """S3-compatible durable source storage (MinIO in Compose)."""

    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url or None,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=Config(signature_version="s3v4"),
        )

    async def setup(self) -> None:
        def ensure() -> None:
            try:
                self.client.head_bucket(Bucket=self.bucket)
            except self.client.exceptions.ClientError:
                self.client.create_bucket(Bucket=self.bucket)

        await asyncio.to_thread(ensure)

    async def put(self, key: str, content: bytes, content_type: str | None = None) -> None:
        kwargs = {"Bucket": self.bucket, "Key": key, "Body": content}
        if content_type:
            kwargs["ContentType"] = content_type
        await asyncio.to_thread(self.client.put_object, **kwargs)

    async def download(self, key: str, destination: Path) -> None:
        await asyncio.to_thread(self.client.download_file, self.bucket, key, str(destination))

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.client.delete_object, Bucket=self.bucket, Key=key)
