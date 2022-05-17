#!/usr/bin/env python
# coding: utf-8

import os
import sys
from glob import glob
from pathlib import Path

import pymongo
import torch
from minio import Minio


def keyboard_interrupt_handler(sig: int, _) -> None:
    minio_leftovers = glob('*.part.minio')
    for leftover in minio_leftovers:
        Path(leftover).unlink()
    sys.exit(1)


def create_s3_client():
    return Minio(os.environ['S3_ENDPOINT'],
                 access_key=os.environ['S3_ACCESS_KEY'],
                 secret_key=os.environ['S3_SECRET_KEY'],
                 region=os.environ['S3_REGION'])


def create_mongodb_client():
    return pymongo.MongoClient(os.environ['DB_CONNECTION_STRING'])


def get_latest_model_weights(s3_client, mongodb_client, skip_download=False):
    db = mongodb_client[os.environ['DB_NAME']]
    latest_model_ts = max(db.model.find().distinct('added_on'))
    model_document = db.model.find_one({'added_on': latest_model_ts})
    model_version = model_document['version']
    model_object_name = f'{model_version}.pt'
    if skip_download:
        return model_version, model_object_name

    weights_url = s3_client.fget_object('model', model_object_name,
                                        model_object_name)
    assert Path(model_object_name).exists()

    return model_version, model_object_name


def init_model(s3, db):
    model_version, model_weights = get_latest_model_weights(s3,
                                                            db,
                                                            skip_download=True)

    if not Path(model_weights).exists():
        model_version, model_weights = get_latest_model_weights(s3, db)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights)
    return model_version, model_weights, model
