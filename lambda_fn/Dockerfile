FROM public.ecr.aws/lambda/python:3.12

# Install inference dependencies first so this layer is cached
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --timeout 200 --retries 10 -r requirements.txt

# Handler code and the trained model
COPY app.py ${LAMBDA_TASK_ROOT}/
COPY model/ ${LAMBDA_TASK_ROOT}/model/

CMD ["app.lambda_handler"]