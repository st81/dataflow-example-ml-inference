# GCP 公式のサンプルコードを参考に作成。
# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/ff104173e276f80a1068ddbdf9d809854d261782/dataflow/flex-templates/pipeline_with_dependencies/Dockerfile

FROM python:3.11-slim as builder

RUN pip install poetry==1.8.5
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt -o requirements.txt

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY --from=apache/beam_python3.11_sdk:2.54.0 /opt/apache/beam /opt/apache/beam

COPY --from=gcr.io/dataflow-templates-base/python311-template-launcher-base:20230622_RC00 /opt/google/dataflow/python_template_launcher /opt/google/dataflow/python_template_launcher

# TODO: Verify if using the Python SDK instead of gsutil reduces the image size.
COPY --from=gcr.io/google.com/cloudsdktool/google-cloud-cli:slim /usr/bin/gcloud /usr/bin/
COPY --from=gcr.io/google.com/cloudsdktool/google-cloud-cli:slim /usr/lib/google-cloud-sdk/ /usr

ARG WORKDIR=/template
WORKDIR ${WORKDIR}

COPY --from=builder /requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY my_package my_package
COPY main.py .

RUN pip install -e .

ENV FLEX_TEMPLATE_PYTHON_PY_FILE=${WORKDIR}/main.py

ENTRYPOINT ["/opt/apache/beam/boot"]