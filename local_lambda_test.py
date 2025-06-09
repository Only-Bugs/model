"""
local_lambda_test.py

A local test harness to simulate AWS Lambda S3 event invocation for the BirdTag project.

This script allows developers to test the `lambda_handler()` locally with mocked
S3 events for different media types: audio, image, or video.

Usage:
    python local_lambda_test.py

Modify the `test_event` payload to match the desired test file in your bucket.

Note:
- Assumes credentials are set up via AWS CLI or environment variables.
- Uses a dummy Lambda context.
"""

import sys
import os

# Ensure current directory is in the module search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lamda.lambda_function import lambda_handler

# --- SELECT ONE TEST EVENT BELOW --- #

# Test: AUDIO file inference (Uncomment to use)
# test_event = {
#     "Records": [
#         {
#             "s3": {
#                 "bucket": {"name": "birdtag-data-bucket"},
#                 "object": {"key": "uploads/sample.wav"}
#             }
#         }
#     ]
# }

# Test: IMAGE file inference (Active test)
test_event = {
    "Records": [
        {
            "s3": {
                "bucket": {"name": "birdtag-data-bucket"},
                "object": {"key": "uploads/kingfisher_3.jpg"}
            }
        }
    ]
}

# Test: VIDEO file inference (Uncomment to use)
# test_event = {
#     "Records": [
#         {
#             "s3": {
#                 "bucket": {"name": "birdtag-data-bucket"},
#                 "object": {"key": "uploads/sample.mp4"}
#             }
#         }
#     ]
# }

# --- Dummy AWS Lambda context (for local use) --- #
class DummyContext:
    """
    A stub LambdaContext object for local testing.

    Attributes are not used in current logic but required by the Lambda signature.
    """
    def __init__(self):
        self.function_name = "local_test"
        self.memory_limit_in_mb = 128
        self.invoked_function_arn = "arn:aws:lambda:dummy"
        self.aws_request_id = "dummy-request-id"

# --- Run the handler --- #
if __name__ == "__main__":
    print("[TEST] Starting local Lambda test...")
    response = lambda_handler(test_event, DummyContext())
    print("[TEST] Lambda Response:", response)
