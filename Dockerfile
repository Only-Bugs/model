FROM python:3.11-slim

WORKDIR /var/task

RUN apt-get update && apt-get install -y --no-install-recommends \
 ffmpeg \
 libsndfile1 \
 libgl1 \
 libglib2.0-0 \
 libsm6 \
 libxext6 \
 libxrender-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/\*

# Install Python dependencies

COPY requirements.txt .



RUN pip install --no-cache-dir awslambdaric \
 && pip install --no-cache-dir --prefer-binary numpy \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy project files

COPY lamda/ lamda/
COPY audio_detection/ audio_detection/
COPY model.pt .

# Runtime Interface Emulator (optional for local testing)

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/local/bin/aws-lambda-rie
RUN chmod +x /usr/local/bin/aws-lambda-rie

ENTRYPOINT ["/usr/local/bin/python3", "-m", "awslambdaric"]
CMD ["lamda.lambda_function.lambda_handler"]
