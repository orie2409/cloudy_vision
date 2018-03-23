import base64
import requests


def _convert_image_to_base64(image_filename):
    with open(image_filename, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


def call_vision_api(image_filename, api_keys, api_regions):
    api_key = api_keys['eyeem']['api_key']
    api_secret = api_keys['eyeem']['api_secret']
    api_url = "https://vision-api.eyeem.com/v1/analyze"
    api_token_url = "https://vision-api.eyeem.com/v1/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {
        "clientId": api_key,
        "clientSecret": api_secret
    }
    response = requests.post(api_token_url, headers=headers, data=payload)
    access_token = response['access_token']

    base64_image = _convert_image_to_base64(image_filename)
    headers = {
        "Authorization": "Bearer" + access_token,
        "Content-Type": "application/json"
    }
    payload = {
        "requests": [
            {
                "tasks": [{"type": "TAGS"}],
                "image": {"content": base64_image}
            }
        ]
    }

    response = requests.post(api_url, headers=headers, data=payload)
    return response


def get_standardized_result(api_result):
    output = {
        'tags': [],
    }


    return output