import json
import requests


def call_vision_api(image_filename, api_keys, api_regions):
    api_key = api_keys['microsoft']
    post_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v1.0/analyze/"

    image_data = open(image_filename, 'rb').read()
    headers = {"Ocp-Apim-Subscription-Key": api_key,
               "Content-Type": "application/octet-stream"}
    params = {'visualFeatures': 'Tags'}
    response = requests.post(post_url,
                             headers=headers,
                             params=params,
                             data=image_data)
    response.raise_for_status()
    return json.loads(response.text)

# Return a dictionary of tags to their scored values (represented as lists of tuples).
# {
#    'feature_1' : [(element, score), ...],
#    'feature_2' : ...
# }
# E.g.,
# {
#    'tags' : [('throne', 0.95), ('swords', 0.84)],
# }


def get_standardized_result(api_result):
    output = {'tags': [(tag['name'], tag['confidence']) for tag in api_result['tags']]}
    return output
