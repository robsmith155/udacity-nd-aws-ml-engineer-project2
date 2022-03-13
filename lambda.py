# FIRST LAMBDA FUNCTION (SERIALIZE IMAGE)
import json
import boto3
import base64

s3 = boto3.resource('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    KEY = event["s3_key"]
    BUCKET_NAME = event["s3_bucket"]

    # Download the data from s3 to /tmp/image.png
    s3.Bucket(BUCKET_NAME).download_file(KEY, '/tmp/image.png')

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": BUCKET_NAME,
            "s3_key": KEY,
            "inferences": []
        }
    }

        
# SECOND LAMBDA FUNCTION (IMAGE CLASSIFIER)

import json
import base64
import boto3

# Fill this in with the name of your deployed model
ENDPOINT_NAME = 'bike-motorbike-image-classifier'

runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Make a prediction
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='image/png', Body=image)
    inferences = json.loads(response['Body'].read().decode('utf-8'))
   
    return {
        'statusCode': 200,
        'body': {
            "image_data": event['body']['image_data'],
            "s3_bucket": event['body']['s3_bucket'],
            "s3_key": event['body']['s3_key'],
            "inferences": inferences
        }
        
# THIRD LAMBDA FUNCTION (CHECK FOR LOW CONFIDENCE PREDICTIONS)

import json


THRESHOLD = .93


def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event['body']['inferences']

    # Check if any values in our inferences are above THRESHOLD
    #meets_threshold = ## TODO: fill in
    meets_threshold = any(x > THRESHOLD for x in inferences) 

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
         'statusCode': 200,
         'body': {
            "image_data": event['body']['image_data'],
            "s3_bucket": event['body']['s3_bucket'],
            "s3_key": event['body']['s3_key'],
            "inferences": event['body']['inferences']
            }
    }