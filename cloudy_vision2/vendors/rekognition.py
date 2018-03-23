import boto3


def call_vision_api(image_filename, api_keys, api_regions):
    api_region = api_regions['rekognition']
    api_key = api_keys['rekognition']['api_key']
    api_secret = api_keys['rekognition']['api_secret']

    client = boto3.client('rekognition',
                          api_region,
                          aws_access_key_id=api_key,
                          aws_secret_access_key=api_secret)

    with open(image_filename, 'rb') as image:
        response = client.detect_labels(Image={'Bytes': image.read()})
        return response


def get_standardized_result(api_result):
    output = {
        'tags': [],
    }
    if 'Labels' not in api_result:
        return output

    labels = api_result['Labels']
    for tag in labels:
        output['tags'].append((tag['Name'], tag['Confidence']/100))

    return output
