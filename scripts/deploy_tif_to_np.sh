#####################################################################################################
# This script deploys a cloud function for converting newly exported files to numpy
#####################################################################################################
export EE_BUCKET=$(python -c "from presto.dataops import EE_BUCKET; print(EE_BUCKET)")
gcloud functions deploy tif-to-np \
    --source=presto \
    --trigger-bucket=$EE_BUCKET \
    --allow-unauthenticated \
    --runtime=python39 \
    --entry-point=trigger_tif_to_np \
    --memory=2048MB \
    --region=us-central1
