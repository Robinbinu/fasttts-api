# FastTTS API - Cloud Run Deployment

This FastAPI application provides Text-to-Speech services and is ready to deploy on Google Cloud Run.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed ([Install Guide](https://cloud.google.com/sdk/docs/install))
3. **Docker** installed (for local testing)

## Quick Deployment

### Option 1: Using the Deployment Script (Recommended)

```bash
# Make the script executable (already done)
chmod +x deploy.sh

# Deploy to Cloud Run
./deploy.sh YOUR_PROJECT_ID us-central1 fasttts-api
```

Replace `YOUR_PROJECT_ID` with your actual GCP project ID.

### Option 2: Manual Deployment

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com

# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fasttts-api
gcloud run deploy fasttts-api \
  --image gcr.io/YOUR_PROJECT_ID/fasttts-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

## Configuration

### Environment Variables

If you need to add API keys for engines like Azure, OpenAI, or ElevenLabs:

```bash
gcloud run services update fasttts-api \
  --region us-central1 \
  --set-env-vars="AZURE_SPEECH_KEY=your-key,AZURE_SPEECH_REGION=your-region"
```

### Resource Limits

The default configuration:
- **Memory**: 2GB
- **CPU**: 2 vCPU
- **Timeout**: 300 seconds
- **Concurrency**: 80 requests per instance
- **Max Instances**: 10

Adjust these in [deploy.sh](deploy.sh) or [cloudbuild.yaml](cloudbuild.yaml) based on your needs.

## Testing Your Deployment

Once deployed, you'll receive a URL like: `https://fasttts-api-xxxxx-uc.a.run.app`

### Test the API:

```bash
# Get available engines
curl https://your-service-url.run.app/engines

# Get available voices
curl https://your-service-url.run.app/voices

# Generate speech
curl 'https://your-service-url.run.app/tts?text=Hello%20World' --output test.wav

# Open the web interface
open https://your-service-url.run.app
```

## CI/CD with Cloud Build

For automatic deployments from GitHub:

1. Connect your repository to Cloud Build
2. Create a trigger using [cloudbuild.yaml](cloudbuild.yaml)
3. Pushes to main will auto-deploy

## Local Testing

Test the Docker image locally before deploying:

```bash
# Build the image
docker build -t fasttts-api .

# Run locally
docker run -p 8080:8080 fasttts-api

# Test
curl http://localhost:8080/engines
```

## Cost Optimization

- Cloud Run charges only for actual usage
- First 2 million requests per month are free
- Adjust `--max-instances` to control costs
- Consider setting `--min-instances=0` for infrequent use

## Supported TTS Engines

Currently configured:
- **Kokoro** (default) - Free, open-source
- Azure, OpenAI, ElevenLabs (require API keys)
- System, Coqui (available but commented out)

Enable additional engines by uncommenting them in `async_server.py` and setting environment variables.

## Troubleshooting

### View logs:
```bash
gcloud run services logs read fasttts-api --region us-central1
```

### Update service:
```bash
gcloud run services update fasttts-api --region us-central1 --memory 4Gi
```

### Delete service:
```bash
gcloud run services delete fasttts-api --region us-central1
```

## Security Notes

- The service is deployed with `--allow-unauthenticated` by default
- For production, consider adding authentication
- Set up Cloud Armor for DDoS protection
- Use Secret Manager for sensitive API keys

## Support

For issues or questions:
- Check Cloud Run logs
- Review FastAPI docs: https://fastapi.tiangolo.com
- Cloud Run docs: https://cloud.google.com/run/docs
