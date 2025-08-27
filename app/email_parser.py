# email_parser.py
import re
import html
import logging
from typing import Dict, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailParser:
    """
    Parses email content from Mailgun webhooks
    """
    
    def __init__(self):
        """Initialize the EmailParser"""
        logger.info("EmailParser initialized")
    
    def parse_mailgun_payload(self, payload: Union[Dict[str, Any], Any]) -> Optional[Dict[str, Any]]:
        """
        Parse Mailgun webhook payload to extract email content
        
        Args:
            payload: Mailgun webhook payload (form data or JSON)
            
        Returns:
            Dict: Parsed email data or None if parsing fails
        """
        try:
            payload_dict = dict(payload) if hasattr(payload, 'items') else payload
            logger.info("Processing Mailgun payload")
            
            email_data = {
                "sender": self._extract_sender(payload_dict),
                "subject": self._extract_subject(payload_dict),
                "text": self._extract_text_content(payload_dict),
                "timestamp": self._extract_timestamp(payload_dict)
            }
            
            if not email_data["text"] or len(email_data["text"].strip()) < 5:
                logger.warning("No meaningful text content found in email")
                return None
            
            logger.info(f"Email parsed successfully from {email_data['sender']}")
            return email_data
        except Exception as e:
            logger.error(f"Error parsing Mailgun payload: {str(e)}")
            return None
    
    def _extract_sender(self, payload: Dict[str, Any]) -> str:
        """Extract sender information"""
        try:
            sender_fields = ['sender', 'from', 'From']
            for field in sender_fields:
                if field in payload:
                    sender = str(payload[field])
                    if sender:
                        return self._clean_email_address(sender)
            return "unknown@unknown.com"
        except Exception as e:
            logger.error(f"Error extracting sender: {str(e)}")
            return "unknown@unknown.com"
    
    def _extract_subject(self, payload: Dict[str, Any]) -> str:
        """Extract email subject"""
        try:
            subject_fields = ['subject', 'Subject']
            for field in subject_fields:
                if field in payload:
                    subject = str(payload[field])
                    if subject:
                        return self._clean_text(subject)
            return "No Subject"
        except Exception as e:
            logger.error(f"Error extracting subject: {str(e)}")
            return "No Subject"
    
    def _extract_text_content(self, payload: Dict[str, Any]) -> str:
        """Extract and combine text content from email"""
        try:
            text_content = ""
            content_fields = ['body-plain', 'body-html', 'stripped-text', 'text']
            for field in content_fields:
                if field in payload:
                    content = str(payload[field])
                    if content and content.strip():
                        text_content = self._html_to_text(content) if 'html' in field.lower() else content
                        break
            if not text_content.strip():
                subject = self._extract_subject(payload)
                if subject and subject != "No Subject":
                    text_content = f"Subject: {subject}"
            return self._clean_text(text_content)
        except Exception as e:
            logger.error(f"Error extracting text content: {str(e)}")
            return ""
    
    def _extract_timestamp(self, payload: Dict[str, Any]) -> str:
        """Extract timestamp from payload"""
        try:
            timestamp_fields = ['timestamp', 'date', 'Date']
            for field in timestamp_fields:
                if field in payload:
                    timestamp_val = payload[field]
                    if isinstance(timestamp_val, (int, float)):
                        return datetime.fromtimestamp(timestamp_val).isoformat()
                    elif isinstance(timestamp_val, str):
                        return self._parse_timestamp_string(timestamp_val)
            return datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Error extracting timestamp: {str(e)}")
            return datetime.now().isoformat()
    
    def _clean_email_address(self, email_addr: str) -> str:
        """Clean and validate email address"""
        try:
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_addr)
            return email_match.group().lower() if email_match else email_addr.strip().lower()
        except Exception as e:
            logger.error(f"Error cleaning email address: {str(e)}")
            return email_addr
    
    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        try:
            if not text:
                return ""
            text = html.unescape(text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return str(text)
    
    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML content to plain text"""
        try:
            text = re.sub(r'<[^>]+>', ' ', html_content)
            text = html.unescape(text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error converting HTML to text: {str(e)}")
            return html_content
    
    def _parse_timestamp_string(self, timestamp_str: str) -> str:
        """Parse various timestamp string formats"""
        try:
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%a, %d %b %Y %H:%M:%S %z'
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str.strip(), fmt)
                    return dt.isoformat()
                except ValueError:
                    continue
            return timestamp_str
        except Exception as e:
            logger.error(f"Error parsing timestamp string: {str(e)}")
            return timestamp_str