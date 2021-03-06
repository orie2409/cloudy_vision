import cloudsight


def call_vision_api(image_filename, api_keys, api_regions):
    api_key = api_keys['cloudsight']
    # api_secret = api_keys['cloudsight']['api_secret']

    # Via example found here:
    # https://github.com/cloudsight/cloudsight-python

    auth = cloudsight.SimpleAuth(api_key)
    api = cloudsight.API(auth)

    with open(image_filename, 'rb') as image_file:
        response = api.image_request(image_file, image_filename)

    response = api.wait(response['token'], timeout=60)
    
    return response


def get_standardized_result(api_result):
    output = {
        'tags': [],
    }

    if api_result['status'] == 'completed':
        output['tags'].append((api_result["name"], None))
    elif api_result['status'] == 'skipped':
        output['tags'].append(("error_skipped_because_" + api_result["reason"], None))
    else:
        output['tags'].append(("error_" + api_result["status"], None))

    return output
