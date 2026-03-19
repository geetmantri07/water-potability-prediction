from dotenv import load_dotenv
import boto3
import os

load_dotenv(override=True)

print("ACCESS:", os.getenv("AWS_ACCESS_KEY_ID"))
print("SECRET:", os.getenv("AWS_SECRET_ACCESS_KEY"))
print("REGION:", os.getenv("AWS_REGION"))

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

print(s3.list_buckets())