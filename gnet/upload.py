"""Uploads all Python modules in this folder to an S3 bucket, to access when
training on Colab or another cloud service.
"""

import os

import boto3

BUCKET_NAME = 'gestalt-graph'
BACKUP_ROOT = os.path.dirname(os.path.abspath(__file__))
db = boto3.resource('s3').Bucket(BUCKET_NAME)

for file in os.listdir(BACKUP_ROOT):
  if file.endswith('.py'):
    with open(os.path.join(BACKUP_ROOT, file), 'r') as f:
      py_code = f.read()
    print(f"Uploading {file}")
    db.put_object(Key=file, Body=py_code)
