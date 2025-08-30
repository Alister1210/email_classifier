# Email Classification API

A FastAPI-based email classification service using a fine-tuned DistilBERT model. This API can classify emails received via Mailgun webhooks or direct text input.

## Features

- 🚀 Fast email classification using DistilBERT
- 📧 Mailgun webhook integration
- 🔄 Batch processing support
- 📊 Confidence scoring and top-K predictions
- 🔍 Comprehensive text preprocessing
- 📝 Detailed logging and monitoring
- ⚡ GPU support for faster inference

## Project Structure

```
email-bot-classifier/
├── app/
│   ├── main.py                 # FastAPI application and routes
│   ├── model_loader.py         # Model and tokenizer loading
│   ├── classify.py             # Classification logic
│   ├── email_parser.py         # Email content extraction
│   └── utils.py                # Helper functions (optional)
├── model/
│   └── fast_model/             # Trained model files
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

## Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd email-bot-classifier
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Place your trained model**
   - Copy the `fast_model` folder from your training script output to `model/fast_model/`
   - Ensure it contains: `config.json`, `pytorch_model.bin`, `tokenizer.json`, etc.

## Usage

### Starting the API Server

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

#### Health Check

```bash
GET /
GET /health
```

#### Mailgun Webhook

```bash
POST /webhook/mailgun
Content-Type: application/x-www-form-urlencoded
# Mailgun sends form data with email content
```

#### Direct Text Classification

```bash
POST /classify
Content-Type: application/json

{
  "text": "Your email content here..."
}
```

#### Model Information

```bash
GET /model/info
```

### Example Usage

**Direct classification:**

```python
import requests

response = requests.post(
    "http://localhost:8000/classify",
    json={"text": "Congratulations! You've won $1000! Click here to claim now."}
)

print(response.json())
# Output:
# {
#   "status": "success",
#   "classification": {
#     "predicted_label": "spam",
#     "confidence": 0.9234,
#     "confidence_category": "very_high"
#   }
# }
```

**Mailgun webhook response:**

```json
{
  "status": "success",
  "email_info": {
    "sender": "user@example.com",
    "subject": "Important notification",
    "timestamp": "2024-01-15T10:30:00"
  },
  "classification": {
    "predicted_label": "ham",
    "confidence": 0.8567,
    "confidence_category": "high"
  },
  "message": "Email classified as: ham"
}
```

## Mailgun Integration

### Setting up Mailgun Webhook

1. **Login to Mailgun Dashboard**
2. **Go to Webhooks section**
3. **Add webhook URL:** `https://your-domain.com/webhook/mailgun`
4. **Select events:** Choose `delivered`, `opened`, etc., based on your needs
5. **Add webhook signing key to your `.env` file**

### Webhook Security

The API can validate Mailgun webhook signatures for security:

```python
# In your .env file
MAILGUN_WEBHOOK_SIGNING_KEY=your_webhook_signing_key
```

## Model Training

The API uses a model trained with the provided training script. Key details:

- **Model:** DistilBERT-base-uncased
- **Max sequence length:** 256 tokens
- **Preprocessing:** Lowercase, remove URLs/emails/numbers, remove punctuation
- **Labels:** Top 3 most frequent classes from your training data

### Important Notes

⚠️ **Label Encoder:** The current implementation creates a default label encoder. For production:

1. Save your label encoder during training:

   ```python
   # Add this to your training script
   import pickle
   with open('./fast_model/label_encoder.pkl', 'wb') as f:
       pickle.dump(label_encoder, f)
   ```

2. Or create a JSON mapping:
   ```python
   import json
   label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
   with open('./fast_model/label_mapping.json', 'w') as f:
       json.dump(label_mapping, f)
   ```

## Configuration

### Environment Variables

| Variable          | Description           | Default            |
| ----------------- | --------------------- | ------------------ |
| `PORT`            | API server port       | 8000               |
| `MODEL_PATH`      | Path to trained model | ./model/fast_model |
| `MAILGUN_API_KEY` | Mailgun API key       | -                  |
| `MAILGUN_DOMAIN`  | Mailgun domain        | -                  |
| `LOG_LEVEL`       | Logging level         | INFO               |

### Model Configuration

The classifier automatically detects:

- GPU availability (CUDA)
- Model architecture
- Number of labels
- Tokenizer settings

## Performance

### Optimization Features

- **GPU acceleration** when available
- **Batch processing** for multiple emails
- **Efficient tokenization** with padding/truncation
- **Memory optimization** with proper tensor management
- **Caching** of model components

### Expected Performance

- **Inference time:** ~50-200ms per email (CPU)
- **Throughput:** ~100-500 emails/minute (depends on hardware)
- **Memory usage:** ~1-2GB RAM (model + overhead)

## Monitoring and Logging

The API provides comprehensive logging:

```python
# Logs include:
- Request/response details
- Classification results
- Error handling
- Performance metrics
```

## Error Handling

The API handles various error scenarios:

- **Invalid input:** Returns 400 with error message
- **Model loading failures:** Returns 503 with details
- **Classification errors:** Returns 500 with error info
- **Empty/malformed emails:** Returns appropriate error codes

## Testing

Run tests (if implemented):

```bash
pytest tests/
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Use ASGI server:** Gunicorn with Uvicorn workers
2. **Add authentication:** API keys or OAuth
3. **Rate limiting:** Prevent abuse
4. **Database logging:** Store classifications for analysis
5. **Monitoring:** Health checks, metrics collection
6. **Load balancing:** Multiple instances for scalability

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

[Your License Here]

## Support

For issues and questions:

- Create GitHub issue
- Contact: [your-email@domain.com]
