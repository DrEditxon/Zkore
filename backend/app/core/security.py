import logging
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from app.core.config import settings

logger = logging.getLogger(__name__)

# Header for programmatic API access
API_KEY_NAME = "X-Zkore-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    """
    Dependency to enforce API Key authentication.
    If ENFORCE_API_KEY is enabled in the environment, all requests to the API
    must include the X-Zkore-API-Key header, OR originate from a trusted IP/Origin.
    """
    if not settings.ENFORCE_API_KEY:
        return api_key

    # Permit if the API key matches
    if api_key and api_key == settings.ZKORE_API_KEY:
        return api_key

    # Check IP Whitelist for trusted internal services
    client_ip = request.client.host if request.client else ""
    if client_ip in ("127.0.0.1", "::1", "localhost"):
        # For local development, we can bypass or enforce. 
        # Here we allow local host bypassing if there's no reverse proxy.
        pass
    else:
        logger.warning(f"[Security] Unauthorized API access attempt from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")

    return api_key


class SecurityHeadersMiddleware:
    """
    FastAPI Middleware that injects Enterprise-grade HTTP Security Headers
    to protect against XSS, clickjacking, MIME-sniffing, and other web vulnerabilities.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                
                # Prevent MIME-sniffing (forces browser to respect Content-Type)
                headers.append((b"X-Content-Type-Options", b"nosniff"))
                
                # Prevent Clickjacking (prevents the app from being embedded in an iframe)
                headers.append((b"X-Frame-Options", b"DENY"))
                
                # Cross-Site Scripting (XSS) Protection filter for legacy browsers
                headers.append((b"X-XSS-Protection", b"1; mode=block"))
                
                # Strict Transport Security (HSTS) - forces HTTPS
                headers.append((b"Strict-Transport-Security", b"max-age=31536000; includeSubDomains"))
                
                # Content Security Policy (CSP)
                # Restricts where scripts, styles, and images can be loaded from.
                csp = (
                    b"default-src 'self'; "
                    b"script-src 'self' 'unsafe-inline'; "
                    b"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                    b"font-src 'self' https://fonts.gstatic.com; "
                    b"img-src 'self' data: https:; "
                    b"connect-src 'self' https:;"
                )
                headers.append((b"Content-Security-Policy", csp))
                
                # Referrer Policy
                headers.append((b"Referrer-Policy", b"strict-origin-when-cross-origin"))

            await send(message)

        await self.app(scope, receive, send_wrapper)
