import datetime
import io
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Union

import apache_beam as beam
import numpy as np
import pandas as pd
import torch
from apache_beam.io.gcp.gcsio import GcsIO
from apache_beam.options.pipeline_options import (
    GoogleCloudOptions,
    PipelineOptions,
    SetupOptions,
)
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from my_package.model import MNISTDataset, MNISTModel

DATASET_CSV_PATH = "/tmp/dataset.csv"
MODEL_CHECKPOINT_PATH = "/tmp/model.pth"


# TODO: Add type hints
def split_list_to_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# TODO: Add type hints
def load_dataset_from_csv(csv_path: str):
    # TODO: Add Docstring

    # TODO: Verify if using the Python SDK instead of gsutil reduces the image size.
    os.system(f"gsutil cp {csv_path} {DATASET_CSV_PATH}")
    df = pd.read_csv(DATASET_CSV_PATH)
    print(df)
    data = df.to_dict(orient="records")
    # TODO: Make chunk size as command line argument.
    # Note that current value is determined by just intuition.
    return list(split_list_to_chunks(data, 10))


# TODO: Add type hints
def _read_image(img_path: str) -> Image:
    with GcsIO().open(img_path, "r") as f:
        return Image.open(io.BytesIO(f.read()))


# TODO: Add type hints
def read_images(data: List[Dict[str, Union[str, int]]], img_root: str):
    # TODO: Add Docstring
    img_data_chunk = []
    for d in data:
        img_path = f"{img_root}/{Path(d['img_path']).name}"
        img = _read_image(img_path)
        img_data_chunk.append({"filename": Path(d["img_path"]).name, "img": img, "target": d["target"]})
    return img_data_chunk


# TODO: Add type hints
def predict(img_data_chunk, device: str):
    # TODO: Add Docstring

    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif device == "cpu":
        logging.warning("Using CPU.")
    else:
        raise RuntimeError("No GPUs found.")

    model.eval()
    model.to(args.device)

    imgs = [d["img"] for d in img_data_chunk]
    targets = [d["target"] for d in img_data_chunk]

    # TODO: move transform to a my_package module
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081))])
    test_dataset = MNISTDataset(imgs, targets, transform)
    # TODO: Make batch size a command line argument.
    dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    all_preds = []
    for batch in dataloader:
        x, _ = batch
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds)
    preds_chunk = [
        {
            "img_filename": d["filename"],
            "target": d["target"],
            "prediction": int(pred),
            "created_at": datetime.datetime.now(),
        }
        for d, pred in zip(img_data_chunk, all_preds)
    ]
    return preds_chunk


# TODO: Add type hints
def run(args, beam_args):
    pipeline_options = PipelineOptions(beam_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # TODO: Confirm if this is necessary
    os.environ["GCLOUD_PROJECT"] = pipeline_options.view_as(GoogleCloudOptions).project

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | beam.Create(load_dataset_from_csv(args.dataset_csv_path))
            | beam.Map(read_images, args.dataset_img_root)
            | beam.Map(predict, args.device)
            | beam.FlatMap(lambda x: x)
            | beam.io.WriteToBigQuery(
                table=args.output_table_id,
                # TODO: Pass schema as command line argument.
                schema="img_filename:STRING,target:INTEGER,prediction:INTEGER,created_at:TIMESTAMP",
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            )
        )
    p.run()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_csv_path", type=str, required=True, help="Path to the dataset CSV. Must be a GCS path."
    )
    parser.add_argument(
        "--dataset_img_root", type=str, required=True, help="Root directory of the dataset images. Must be a GCS path."
    )
    parser.add_argument(
        "--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint. Must be a GCS path."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    parser.add_argument("--output_table_id", type=str, required=True, help="BigQuery output table ID.")
    args, beam_args = parser.parse_known_args()

    # TODO: Make error message more informative
    assert (
        args.dataset_csv_path.startswith("gs://")
        and args.dataset_img_root.startswith("gs://")
        and args.model_checkpoint_path.startswith("gs://")
    ), "Paths must be GCS paths."

    # TODO: Remove this
    print(f"cuda available: {torch.cuda.is_available()}")

    # TODO: Verify if using the Python SDK instead of gsutil reduces the image size.
    os.system(f"gsutil cp {args.model_checkpoint_path} {MODEL_CHECKPOINT_PATH}")
    # Load on CPU because worker to launch Flex Template don't have GPU.
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]
    # Make `model` a global variable to avoid load model weights in each worker.
    model = MNISTModel(hidden_size=hparams["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"])

    run(args, beam_args)
