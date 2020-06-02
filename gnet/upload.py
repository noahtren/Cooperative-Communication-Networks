"""Running this uploads all Python modules in this folder to an S3 bucket, to
access when training on Colab or another cloud service.
"""

import os

import boto3

from cfg import CFG

BUCKET_NAME = CFG['s3_bucket']
code_path = os.path.dirname(os.path.abspath(__file__))
db = boto3.resource('s3').Bucket(BUCKET_NAME)
s3_client = boto3.client('s3')


def upload_data(s3_path, body):
  """Upload data to S3 bucket at a particular path
  """
  db.put_object(Key=s3_path, Body=body)


def upload_file(local_path, s3_path):
  """Upload a locally-saved file to S3
  """
  s3_client.upload_file(local_path, BUCKET_NAME, s3_path)


if __name__ == "__main__":
  for file in os.listdir(code_path):
    if file.endswith('.py') or file.endswith('.json'):
      with open(os.path.join(code_path, file), 'r') as f:
        py_code = f.read()
      print(f"Uploading {file}")
      upload_data(f"code/{file}", py_code)
