FROM apache/airflow:3.3.2

# Extra packages for the ML pipeline.
# apache-airflow is pinned to the image's own version so pip cannot change it.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --timeout 200 --retries 10 \
    "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt