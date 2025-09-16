FROM nvcr.io/nvidia/tritonserver:25.08-py3-sdk

COPY benchmarks/ ./benchmarks/

COPY deploy/ ./deploy/

RUN pip install -r ./deploy/utils/requirements.txt

ENTRYPOINT ["python", "-m", "benchmarks.utils.benchmark"]
