import datetime
import json
import numpy as np
import os
import pandas as pd
from shutil import copyfile
import time

from jinja2 import FileSystemLoader, Environment
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import SpaceTokenizer

from caltech_image_processing import extract_and_relabel_images, create_image_tags

import vendors.google
import vendors.microsoft
import vendors.clarifai_
import vendors.ibm
import vendors.cloudsight_
import vendors.rekognition


SETTINGS = None


def settings(name):
    """Fetch a settings parameter."""

    # Initialize settings if necessary.
    global SETTINGS
    if SETTINGS is None:

        # Change this dict to suit your taste.
        SETTINGS = {
            'api_keys_filepath': './api_keys.json',
            'api_regions_filepath': './api_regions.json',
            'input_images_dir': 'input_images',
            'output_dir': 'output',
            'static_dir': 'static',
            'output_image_height': 200,
            'vendors': {
                'google': vendors.google,
                'msft': vendors.microsoft,
                'clarifai': vendors.clarifai_,
                'ibm': vendors.ibm,
                'cloudsight': vendors.cloudsight_,
                'rekognition': vendors.rekognition
            },
            'resize': False,
            'statistics': [
                'response_time',
                'tags_count',
            ],
            'tagged_images': True,
            'tags_filepath': './tags.json',
        }

        if SETTINGS['tagged_images']:
            SETTINGS['statistics'] += [
                'matching_tags_count',
                'matching_confidence'
            ]

        # Load API keys
        with open(SETTINGS['api_keys_filepath']) as data_file:
            SETTINGS['api_keys'] = json.load(data_file)

        # Load API Regions
        with open(SETTINGS['api_regions_filepath']) as data_file:
            SETTINGS['api_regions'] = json.load(data_file)

        if settings('resize'):
            from PIL import Image

    return SETTINGS[name]


def log_status(filepath, vendor_name, msg):
    filename = os.path.basename(filepath)
    print("%s -> %s" % ((filename + ", " + vendor_name).ljust(40), msg))


def resize_and_save(input_image_filepath, output_image_filepath):
    image = Image.open(input_image_filepath)
    height = image.size[0]
    width = image.size[1]
    aspect_ratio = float(width) / float(height)

    new_height = settings('output_image_height')
    new_width = int(aspect_ratio * new_height)

    image.thumbnail((new_width, new_height))
    image.save(output_image_filepath)


def render_from_template(directory, template_name, **kwargs):
    loader = FileSystemLoader(directory)
    env = Environment(loader=loader)
    template = env.get_template(template_name)
    return template.render(**kwargs)


def vendor_statistics(image_results):
    vendor_stats = {}

    if len(settings('statistics')) == 0:
        return vendor_stats

    for vendor in settings('vendors'):
        vendor_results = []
        for image_result in image_results:
            for res in image_result['vendors']:
                if res['vendor_name'] == vendor:
                    vendor_results.append(res)

        vendor_stats[vendor] = []
        for stat_key in settings('statistics'):
            values = np.array([vr[stat_key] for vr in vendor_results])
            if stat_key == 'matching_confidence':
                # Wish to consider confidence only for those with a match.
                values = [value for value in values if (value > 0 or value is not None)]

            vendor_stats[vendor].append({
                'name': 'mean_' + stat_key,
                'value': np.average(values)
            })
            vendor_stats[vendor].append({
                'name': 'stdev_' + stat_key,
                'value': np.std(values)
            })

    return vendor_stats


def find_matching_tags(tags, standardized_result):
    matching_tags = set()
    lemmatizer = WordNetLemmatizer()
    tokenizer = SpaceTokenizer()
    for tag in tags:
        # Convert to lowercase and lemmatize
        lemma = lemmatizer.lemmatize(tag.lower())
        for (res_name, res_conf) in standardized_result['tags']:
            # Convert to lowercase, lemmatize and tokenise
            res_name_proc = lemmatizer.lemmatize(res_name.lower())
            res_name_proc = tokenizer.tokenize(res_name_proc)
            # If tag matches any of the words in the api result, then add api result to matching_tags
            if any([res_name_part == lemma for res_name_part in res_name_proc]):
                matching_tags.add((res_name, res_conf))
    return list(matching_tags)


def process_all_images():

    image_results = []

    # Create the output directory
    if not os.path.exists(settings('output_dir')):
        os.makedirs(settings('output_dir'))

    # Read image labels
    if settings('tagged_images'):
        with(open(settings('tags_filepath'), 'r')) as tags_file:
            tags = json.loads(tags_file.read())

    # Loop through all input images.
    for filename in os.listdir(settings('input_images_dir')):

        # Only process files that have these image extensions.
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            continue

        # Create a full path so we can read these files.
        filepath = os.path.join(settings('input_images_dir'), filename)

        # Read desired tags to compare against if specified
        image_tags = []
        if settings('tagged_images'):
            image_tags = tags.get(filename, [])

        # Create an output object for the image
        image_result = {
            'input_image_filepath': filepath,
            'output_image_filepath': filename,
            'vendors': [],
            'image_tags': image_tags,
        }
        image_results.append(image_result)

        # If there's no output file, then resize or copy the input file over
        output_image_filepath = os.path.join(settings('output_dir'), filename)
        if not(os.path.isfile(output_image_filepath)):
            log_status(filepath, "", "writing output image in %s" % output_image_filepath)
            if settings('resize'):
                resize_and_save(filepath, output_image_filepath)
            else:
                copyfile(filepath, output_image_filepath)

        # Walk through all vendor APIs to call.
        for vendor_name, vendor_module in sorted(settings('vendors').items(), reverse=True):

            # Figure out filename to store and retrieve cached JSON results.
            output_json_filename = filename + "." + vendor_name + ".json"
            output_json_path = os.path.join(settings('output_dir'), output_json_filename)

            # Check if the call is already cached.
            if os.path.isfile(output_json_path):

                # If so, read the result from the .json file stored in the output dir.
                log_status(filepath, vendor_name, "skipping API call, already cached")
                with open(output_json_path, 'r') as infile:
                    api_result = json.loads(infile.read())

            else:

                # If not, make the API call for this particular vendor.
                log_status(filepath, vendor_name, "calling API")
                api_call_start = time.time()
                api_result = vendor_module.call_vision_api(filepath, settings('api_keys'), settings('api_regions'))
                api_result['response_time'] = time.time() - api_call_start

                # And cache the result in a .json file
                log_status(filepath, vendor_name, "success, storing result in %s" % output_json_path)
                with open(output_json_path, 'w') as outfile:
                    api_result_str = json.dumps(api_result, sort_keys=True, indent=4, separators=(',', ': '))
                    outfile.write(api_result_str)

                # Sleep so we avoid hitting throttling limits
                time.sleep(1)

            # Parse the JSON result we fetched (via API call or from cache)
            standardized_result = vendor_module.get_standardized_result(api_result)

            # Sort tags if found
            if 'tags' in standardized_result:
                standardized_result['tags'].sort(key=lambda tup: tup[1], reverse=True)

            # If expected tags are provided, calculate accuracy
            tags_count = 0
            matching_tags = []
            matching_confidence = 0
            if 'tags' in standardized_result:
                tags_count = len(standardized_result['tags'])

                if settings('tagged_images'):
                    matching_tags = find_matching_tags(image_tags, standardized_result)
                    matching_scores = [score for (_, score) in matching_tags]
                    if len(matching_tags) > 0 and not any([score is None for score in matching_scores]):
                        matching_confidence = np.mean(matching_scores)

            image_result['vendors'].append({
                'api_result': api_result,
                'vendor_name': vendor_name,
                'standardized_result': standardized_result,
                'output_json_filename': output_json_filename,
                'response_time': api_result['response_time'],
                'tags_count': tags_count,
                'matching_tags': matching_tags,
                'matching_tags_count': len(matching_tags),
                'matching_confidence': matching_confidence,
            })

    # Compute global statistics for each vendor
    vendor_stats = vendor_statistics(image_results)

    # Sort image_results output by filename (so that future runs produce comparable output)
    image_results.sort(key=lambda image_result: image_result['output_image_filepath'])

    # Render HTML file with all results.
    output_html = render_from_template(
        'static',
        'template.html',
        image_results=image_results,
        vendor_stats=vendor_stats,
        process_date=datetime.datetime.today()
    )

    # Write HTML output.
    output_html_filepath = os.path.join(settings('output_dir'), 'output.html')
    with open(output_html_filepath, 'w') as output_html_file:
        output_html_file.write(output_html)
    return image_results


def results_to_csv(image_results, out_filepath, out_filename):

    image_names = []
    image_tags = []
    vendor_names = []
    response_times = []
    tags_counts = []
    matching_tags_counts = []
    matching_confidences = []

    for item in image_results:
        image_name = item['input_image_filepath']
        image_tag = item['image_tags'][0]
        for sub_item in item['vendors']:
            vendor_name = sub_item['vendor_name']
            response_time = sub_item['response_time']
            tags_count = sub_item['tags_count']
            matching_tags_count = sub_item['matching_tags_count']
            matching_confidence = sub_item['matching_confidence']

            image_names.append(image_name)
            image_tags.append(image_tag)
            vendor_names.append(vendor_name)
            response_times.append(response_time)
            tags_counts.append(tags_count)
            matching_tags_counts.append(matching_tags_count)
            matching_confidences.append(matching_confidence)

    df = pd.DataFrame({
        'image_name': image_names,
        'image_tag': image_tags,
        'vendor_name': vendor_names,
        'response_time': response_times,
        'tags_count ': tags_counts,
        'matching_tags_count': matching_tags_counts,
        'matching_confidence': matching_confidences
    })

    if not os.path.exists(out_filepath):
        os.mkdir(out_filepath)

    df.to_csv(os.path.join(out_filepath, out_filename))


if __name__ == "__main__":

    input_path_root = './cloudy_vision/caltech101_images'
    output_path = './cloudy_vision/input_images'
    tags_json_filepath = './cloudy_vision/tags.json'

    out_filepath = '.'
    out_filename = 'image_results_ibm_20180321.csv'

    extract_and_relabel_images(input_path_root, output_path, count=2)
    create_image_tags(output_path, tags_json_filepath)

    output = process_all_images()
    results_to_csv(output, out_filepath=out_filepath, out_filename=out_filename)



