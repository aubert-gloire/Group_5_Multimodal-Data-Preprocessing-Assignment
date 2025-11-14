"""
Small helper script to download model artifacts from a remote URL listed in model-metadata.json.

This is optional but convenient: keep model binaries out of git, publish them to GitHub Releases
or S3, and store the metadata (filename, URL, sha256) in `model-metadata.json` so this script
can fetch them and verify integrity.

Usage:
    python scripts/download_models.py --metadata model-metadata.json --out models

The script will create `models/` and download files listed in the metadata.
"""
import argparse
import json
import os
import hashlib
from urllib.request import urlopen, urlretrieve


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main(metadata_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    for item in meta.get('artifacts', []):
        filename = item['filename']
        url = item['url']
        expected_sha = item.get('sha256')
        target = os.path.join(out_dir, filename)
        if os.path.exists(target):
            if expected_sha and sha256_of_file(target) == expected_sha:
                print(f'{filename} already present and checksum matches')
                continue
            else:
                print(f'{filename} present but checksum mismatch or not provided - re-downloading')
        print('Downloading', url)
        urlretrieve(url, target)
        if expected_sha:
            got = sha256_of_file(target)
            if got != expected_sha:
                raise ValueError(f'Checksum mismatch for {filename}: expected {expected_sha} got {got}')
        print('Saved', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='model-metadata.json')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()
    main(args.metadata, args.out)
