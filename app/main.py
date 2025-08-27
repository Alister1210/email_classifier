from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any
import json
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func
from .model_loader import ModelLoader
from .classify import EmailClassifier
from .email_parser import EmailParser
from datetime import datetime
from pydantic import BaseModel
import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest  # ✅ alias

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification API",
    description="Classify emails using fine-tuned DistilBERT model",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://your-frontend-domain.com",
        "http://localhost:8000",
        "https://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize database
Base = declarative_base()
class Classification(Base):
    __tablename__ = "classifications"
    id = Column(Integer, primary_key=True)
    message_id = Column(String, unique=True, index=True)
    sender = Column(String)
    subject = Column(String)
    label = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///classifications.db")
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine, checkfirst=True)
Session = sessionmaker(bind=engine)

# Global variables
model_loader = None
email_classifier = None
email_parser = None
stored_credentials = None
flow = None

# ✅ replace deprecated startup with lifespan
@app.on_event("startup")
async def startup_event():
    global model_loader, email_classifier, email_parser, flow
    try:
        logger.info("Loading model components...")
        model_path = os.getenv("MODEL_PATH", "./model/email_classifier_20250809_231603")
        model_loader = ModelLoader(model_path)
        model, tokenizer, label_encoder, max_length = model_loader.load_components()
        email_classifier = EmailClassifier(model, tokenizer, label_encoder, max_length)
        email_parser = EmailParser()

        flow = Flow.from_client_secrets_file(
            os.getenv("GOOGLE_CLIENT_SECRETS", "client_secrets.json"),
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            redirect_uri=os.getenv("GOOGLE_REDIRECT_URI", "https://localhost:8000/api/auth/callback"),
        )

        logger.info("Model components and OAuth flow loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}", exc_info=True)
        raise

@app.get("/")
async def root():
    return {"message": "Email Classification API is running", "status": "healthy"}


@app.get("/api/auth/profile")
async def auth_profile():
    global stored_credentials
    try:
        if not stored_credentials:
            raise HTTPException(status_code=401, detail="Not authenticated")

        creds = Credentials.from_authorized_user_info(json.loads(stored_credentials))

        # ✅ safe refresh
        if not creds.valid and creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
            stored_credentials = creds.to_json()

        resp = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            params={"alt": "json"},
            headers={"Authorization": f"Bearer {creds.token}"},
        )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching profile: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    try:
        if not all([model_loader, email_classifier, email_parser]):
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "Components not loaded"},
            )
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)},
        )

@app.post("/api/webhook/mailgun")
@limiter.limit("5/minute")
async def mailgun_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            form_data = await request.form()
            payload = dict(form_data)
        elif "application/json" in content_type:
            payload = await request.json()
        else:
            raise HTTPException(status_code=400, detail="Unsupported content type")

        logger.info("Received Mailgun webhook")
        email_data = email_parser.parse_mailgun_payload(payload)
        if not email_data:
            raise HTTPException(status_code=400, detail="Could not extract email content")

        classification_result = email_classifier.classify_email(email_data["text"])
        response = {
            "status": "success",
            "email_info": {
                "sender": email_data.get("sender", "Unknown"),
                "subject": email_data.get("subject", "No Subject"),
                "timestamp": email_data.get("timestamp"),
            },
            "classification": classification_result,
        }

        background_tasks.add_task(log_classification, email_data, classification_result)
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel

class ClassifyRequest(BaseModel):
    text: str


@app.post("/api/classify")
@limiter.limit("5/minute")
async def classify_text(request: Request, payload: ClassifyRequest):
    try:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")

        result = email_classifier.classify_email(text)
        return JSONResponse(content={"status": "success", "classification": result})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text classification: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/login")
async def auth_login():
    try:
        authorization_url, _ = flow.authorization_url(prompt='consent')
        logger.info("OAuth login initiated")
        return {"authorization_url": authorization_url}
    except Exception as e:
        logger.error(f"Error initiating OAuth: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from google.oauth2.credentials import Credentials

@app.get("/api/auth/callback")
async def auth_callback(request: Request, background_tasks: BackgroundTasks):
    try:
        flow.fetch_token(authorization_response=str(request.url))

        global stored_credentials
        creds = flow.credentials
        stored_credentials = creds.to_json()   # ✅ proper credentials JSON

        logger.info("OAuth callback successful, credentials stored")
        background_tasks.add_task(fetch_and_classify_emails)

        # Redirect to frontend with success
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:5173')
        return RedirectResponse(url=f"{frontend_url}?auth=success")
    except Exception as e:
        logger.error(f"Error in OAuth callback: {str(e)}", exc_info=True)
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:5173')
        return RedirectResponse(url=f"{frontend_url}?auth=error&message={str(e)}")

        
@app.get("/api/results")
async def get_results(limit: int = 50):
    try:
        session = Session()
        classifications = (
            session.query(Classification)
            .order_by(Classification.timestamp.desc())
            .limit(limit)
            .all()
        )
        results = [
            {
                "message_id": c.message_id,
                "sender": c.sender,
                "subject": c.subject,
                "label": c.label,
                "confidence": c.confidence,
                "timestamp": c.timestamp.isoformat(),
            }
            for c in classifications
        ]
        session.close()
        return {"status": "success", "classifications": results}
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class FullEmailsQuery(BaseModel):
    limit: int = 20

@app.post("/api/emails/full")
async def get_full_emails(query: FullEmailsQuery):
    try:
        if not stored_credentials:
            raise HTTPException(status_code=401, detail="Not authorized with Gmail")

        credentials = flow.credentials.from_authorized_user_info(json.loads(stored_credentials))
        service = build('gmail', 'v1', credentials=credentials)

        results = service.users().messages().list(userId='me', maxResults=min(max(query.limit, 1), 50), q='').execute()
        messages = results.get('messages', [])
        emails = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'unknown@unknown.com')
            date_header = next((h['value'] for h in headers if h['name'] == 'Date'), None)
            timestamp = None
            try:
                timestamp = datetime.strptime(date_header, '%a, %d %b %Y %H:%M:%S %z').isoformat() if date_header else datetime.now().isoformat()
            except Exception:
                timestamp = datetime.now().isoformat()

            body = ''
            if 'data' in msg_data['payload'].get('body', {}):
                body = base64.urlsafe_b64decode(msg_data['payload']['body']['data']).decode('utf-8', errors='ignore')
            elif msg_data['payload'].get('parts'):
                for part in msg_data['payload']['parts']:
                    if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        break
                    elif part.get('mimeType') == 'text/html' and part.get('body', {}).get('data'):
                        html_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        body = email_parser._html_to_text(html_body)
                        break

            classification = email_classifier.classify_email(body or (msg_data.get('snippet') or '')) if email_classifier else {"predicted_label": "unknown", "confidence": 0.0}
            emails.append({
                "message_id": msg['id'],
                "sender": sender,
                "subject": subject,
                "timestamp": timestamp,
                "snippet": msg_data.get('snippet', ''),
                "body": body,
                "classification": classification,
            })

        return {"status": "success", "emails": emails}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching full emails: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/debug/db")
async def debug_db():
    try:
        session = Session()
        count = session.query(Classification).count()
        sample = session.query(Classification).order_by(Classification.timestamp.desc()).limit(5).all()
        duplicates = session.query(Classification.message_id, func.count().label('count')).group_by(Classification.message_id).having(func.count() > 1).all()
        session.close()
        return {
            "status": "success",
            "total_rows": count,
            "duplicates": [{"message_id": d.message_id, "count": d.count} for d in duplicates],
            "sample": [
                {
                    "message_id": c.message_id,
                    "sender": c.sender,
                    "subject": c.subject,
                    "label": c.label,
                    "confidence": c.confidence,
                    "timestamp": c.timestamp.isoformat()
                } for c in sample
            ]
        }
    except Exception as e:
        logger.error(f"Error debugging database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cleanup/duplicates")
async def cleanup_duplicates():
    try:
        session = Session()
        duplicates = session.query(Classification.message_id, func.count().label('count')).group_by(Classification.message_id).having(func.count() > 1).all()
        deleted_count = 0
        for dup in duplicates:
            # Keep the latest record by timestamp
            to_delete = session.query(Classification).filter(Classification.message_id == dup.message_id).order_by(Classification.timestamp.desc()).offset(1).all()
            for record in to_delete:
                session.delete(record)
                deleted_count += 1
        session.commit()
        session.close()
        logger.info(f"Cleaned up {deleted_count} duplicate classifications")
        return {"status": "success", "message": f"Deleted {deleted_count} duplicate classifications"}
    except Exception as e:
        logger.error(f"Error cleaning up duplicates: {str(e)}", exc_info=True)
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_and_classify_emails():
    try:
        if not stored_credentials:
            logger.warning("No credentials available for Gmail API")
            return
        session = Session()
        logger.info("Starting email fetch and classification")
        credentials = flow.credentials.from_authorized_user_info(json.loads(stored_credentials))
        service = build('gmail', 'v1', credentials=credentials)
        logger.info("Gmail API service initialized")
        
        results = service.users().messages().list(userId='me', maxResults=10, q='is:unread').execute()
        messages = results.get('messages', [])
        logger.info(f"Fetched {len(messages)} unread messages")
        
        for msg in messages:
            existing = session.query(Classification).filter_by(message_id=msg['id']).first()
            if existing:
                logger.info(f"Skipping already classified message ID {msg['id']} (existing label: {existing.label}, timestamp: {existing.timestamp})")
                continue
            
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'unknown@unknown.com')
            logger.info(f"Processing message ID {msg['id']} from {sender}")
            
            body = ''
            if 'data' in msg_data['payload'].get('body', {}):
                body = base64.urlsafe_b64decode(msg_data['payload']['body']['data']).decode('utf-8', errors='ignore')
            elif msg_data['payload'].get('parts'):
                for part in msg_data['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        break
                    elif part['mimeType'] == 'text/html':
                        html_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                        body = email_parser._html_to_text(html_body)
                        break
            
            logger.info(f"Classifying email with subject: {subject}")
            result = email_classifier.classify_email(body)
            classification = Classification(
                message_id=msg['id'],
                sender=sender,
                subject=subject,
                label=result['predicted_label'],
                confidence=result['confidence'],
                timestamp=datetime.now()
            )
            try:
                session.add(classification)
                session.commit()
                logger.info(f"Stored classification: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
            except IntegrityError as e:
                logger.warning(f"Duplicate message_id {msg['id']} detected, skipping: {str(e)}")
                session.rollback()
        
        logger.info(f"Processed {len(messages)} emails")
    except HttpError as e:
        logger.error(f"Gmail API error: {str(e)}", exc_info=True)
        session.rollback()
    except Exception as e:
        logger.error(f"Error fetching/classifying: {str(e)}", exc_info=True)
        session.rollback()
    finally:
        session.close()

def log_classification(email_data: Dict, classification_result: Dict):
    try:
        log_entry = {
            "sender": email_data.get("sender"),
            "subject": email_data.get("subject"),
            "predicted_label": classification_result["predicted_label"],
            "confidence": classification_result["confidence"],
            "timestamp": email_data.get("timestamp")
        }
        logger.info(f"Classification logged: {json.dumps(log_entry)}")
    except Exception as e:
        logger.error(f"Error logging classification: {str(e)}", exc_info=True)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem"
    )