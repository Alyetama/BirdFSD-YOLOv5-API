#!/usr/bin/env python
# coding: utf-8

import hashlib
import json
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pymongo
import requests
from fic import fic
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Response, UploadFile

from api_utils import create_s3_client, create_mongodb_client, init_model

load_dotenv()

s3 = create_s3_client()
db = create_mongodb_client()
model_version, model_weights, model = init_model(s3, db)

app = FastAPI()


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            # separators=(", ", ": "),
        ).encode("utf-8")


def species_info(species_name):
    if '(' in species_name:
        species_name = species_name.split('(')[1].split(')')[0]
    url = f'https://api.gbif.org/v1/species/match?name={species_name}'
    response = requests.get(url)
    json_resp = response.json()
    if not json_resp.get('usageKey'):
        return 'no results'
    gbif_key = json_resp['usageKey']
    gbif_url = f'https://www.gbif.org/species/{gbif_key}'
    compact_info = {
        k: v
        for k, v in json_resp.items() if 'key' not in k.lower()
        and k not in ['status', 'confidence', 'matchType'] or k == 'usageKey'
    }
    return compact_info


def model_info(version):
    col = db[os.environ['DB_NAME']].model
    if version == 'latest':
        latest_model_ts = max(col.find().distinct('added_on'))
        model = col.find({'added_on': latest_model_ts}).next()
    else:
        model = col.find({'version': version}).next()
    model.pop('_id')
    model['added_on'] = str(model['added_on'])
    model['trained_on'] = str(model['trained_on'])
    model.pop('projects')
    return model


@app.get("/model")
def get_model_info(version: str = 'latest'):
    model = model_info(version)
    return PrettyJSONResponse(status_code=200, content=model)


@app.post("/predict")
def predict_endpoint(file: UploadFile, download: bool = False):
    image = Image.open(file.file)
    content_type = mimetypes.guess_type(file.filename)[0]

    pred = model(image)
    pred_results = pd.concat(pred.pandas().xyxyn).T.to_dict()
    for k in pred_results:
        K = pred_results[k]
        K.pop('class')
        K.update({'confidence': round(K['confidence'], 4)})
        bbox = {}
        for _k in ['xmin', 'ymin', 'xmax', 'ymax']:
            bbox.update({_k: K[_k]})
            K.pop(_k)
        K['bbox'] = bbox
        K['species_info'] = {'gbif': species_info(K['name'])}

    im = Image.fromarray(pred.render()[0])

    with tempfile.NamedTemporaryFile() as f:
        im.save(f, format=content_type.split('/')[1])

        _ = fic.compress(f.name, to_jpeg=True, save_to=f.name)
        f.seek(0)

        if download:
            return Response(f.read(), status_code=200, media_type=content_type)

        length = Path(f.name).stat().st_size
        out_file = f'{str(uuid.uuid4()).split("-")[-1]}.jpg'

        s3.put_object(bucket_name='api',
                      object_name=out_file,
                      data=f,
                      length=length,
                      content_type=content_type)

    url = f'https://{os.environ["S3_ENDPOINT"]}/api/{out_file}'

    if os.getenv('MODEL_REPO'):
        page = f'{os.getenv("MODEL_REPO")}/releases/tag/{model_version}'
    else:
        page = None

    res = {
        'results': {
            'input_image': {'name': file.filename,
            'hash': hashlib.md5(file.file.read()).hexdigest()
            },
            'labeled_image_url': url,
            'predictions': pred_results,
            'model': {
                'name':
                model_version.split('-v')[0],
                'version':
                model_version.split('-v')[1],
                'page': page
            }
        }
    }

    return PrettyJSONResponse(status_code=200, content=res)