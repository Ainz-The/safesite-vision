import json
import logging
import os

import boto3

log = logging.getLogger("aws")

REGION = os.getenv("AWS_REGION", "ap-southeast-2")
S3_BUCKET = os.getenv("S3_BUCKET", "")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "")


def save_run(run_id, original_bytes, annotated_bytes, record):
    if not S3_BUCKET:
        log.info("S3 not configured, skipping save")
        return False
    try:
        s3 = boto3.client("s3", region_name=REGION)
        prefix = f"runs/{run_id}/"
        s3.put_object(Bucket=S3_BUCKET, Key=prefix + "original", Body=original_bytes)
        s3.put_object(Bucket=S3_BUCKET, Key=prefix + "annotated.jpg", Body=annotated_bytes, ContentType="image/jpeg")
        s3.put_object(Bucket=S3_BUCKET, Key=prefix + "trace.json", Body=json.dumps(record, indent=2).encode(), ContentType="application/json")
        return True
    except Exception:
        log.exception("S3 save failed")
        return False


def send_alert(subject, message):
    if not SNS_TOPIC_ARN:
        log.info("SNS not configured, alert not sent")
        return False
    try:
        sns = boto3.client("sns", region_name=REGION)
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message)
        return True
    except Exception:
        log.exception("SNS alert failed")
        return False