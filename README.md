# BirdFSD-YOLOv5 API

[![Supported Python versions](https://img.shields.io/badge/Python-%3E=3.9-blue.svg)](https://www.python.org/downloads/) [![PEP8](https://img.shields.io/badge/Code%20style-PEP%208-orange.svg)](https://www.python.org/dev/peps/pep-0008/) 

## Requirements
- [python>=3.9](https://www.python.org/downloads/)


## Getting started

- Set the required environment variables.

```sh
mv .env.example .env
nano .env  # or any other editor
```

- Install requirements

```sh
pip install -r requirements.txt
```

- Run locally

```sh
uvicorn api:app --reload
```

## Example

```sh
curl -X POST "http://127.0.0.1:8000/predict" -F file="@demo/demo.png"
```

```json
{
    "results": {
        "input_image": {
            "name": "demo.png",
            "hash": "6c9ed1b20b82112e057ac2ece74795ec"
        },
        "labeled_image_url": "https://api-s3.aibird.me/api/fce363ea01c8.jpg",
        "predictions": {
            "0": {
                "confidence": 0.9704,
                "name": "Brown-headed Cowbird",
                "bbox": {
                    "xmin": 0.2905663549900055,
                    "ymin": 0.5605813264846802,
                    "xmax": 0.5082285404205322,
                    "ymax": 0.8753182888031006
                },
                "species_info": {
                    "gbif": "no results"
                }
            }
        },
        "model": {
            "name": "BirdFSD-YOLOv5",
            "version": "1.0.0-alpha.4",
            "page": null
        }
    }
}
```
