#!/bin/bash

ENV_VARS="BUCKET_NAME=$BUCKET_NAME,BUCKET=$BUCKET,INSTANCE=$INSTANCE,MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI,MLFLOW_EXPERIMENT=$MLFLOW_EXPERIMENT,MLFLOW_MODEL_NAME=$MLFLOW_MODEL_NAME,BATCH_SIZE=$BATCH_SIZE,EPOCHS=$EPOCHS,VALIDATION_SPLIT=$VALIDATION_SPLIT,TEST_SPLIT=$TEST_SPLIT,LOCAL_MODEL_PATH=$LOCAL_MODEL_PATH,PRODUCTION_MODEL_NAME=$PRODUCTION_MODEL_NAME,EVALUATION_MODEL_NAME=$EVALUATION_MODEL_NAME,GCP_PROJECT_ID=$GCP_PROJECT_ID,DOCKER_IMAGE_NAME=$DOCKER_IMAGE_NAME,GCP_REGION=$GCP_REGION,DOCKER_REPO_NAME=$DOCKER_REPO_NAME"

docker build --platform linux/amd64 -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$DOCKER_IMAGE_NAME:0.3 .

docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$DOCKER_IMAGE_NAME:0.3

gcloud run deploy --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$DOCKER_IMAGE_NAME:0.3 --region $GCP_REGION --memory=2Gi --set-env-vars $ENV_VARS
