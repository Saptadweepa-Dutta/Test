"""
MCP Server for OpenAPI/Swagger Spec Management with Persistent Storage
"""
import os
import re
import json
import time
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Iterable
from urllib.parse import urlencode
from pathlib import Path

import requests
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from mcp.server.fastmcp import FastMCP
from mcp.server import Context
from mcp.server.session import ServerSession

# Setup logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Optional: Try to import openapi3-parser if available
HAVE_OPENAPI_PARSER = False
try:
    from openapi3_parser import parse as parse_openapi
    HAVE_OPENAPI_PARSER = True
except ImportError:
    pass

# Database configuration
DB_PATH = os.getenv("MCP_DB_PATH", "mcp_swagger_specs.db")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class SpecRecord:
    """Record for storing OpenAPI spec metadata"""
    name: str
    raw: dict
    base_url: Optional[str] = None
    headers: dict = field(default_factory=dict)
    parser: Optional[object] = None  # Optionally hold a parser object


class DatabaseStore:
    """Persistent store of OpenAPI specs using SQLite."""
    
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()
        # In-memory cache for faster access
        self._cache: dict[str, SpecRecord] = {}
        self._load_cache()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS openapi_specs (
                name TEXT PRIMARY KEY,
                raw_spec TEXT NOT NULL,
                base_url TEXT,
                headers TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        log.info(f"Database initialized at {self.db_path}")
    
    def _load_cache(self) -> None:
        """Load all specs from database into memory cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name, raw_spec, base_url, headers FROM openapi_specs")
        rows = cursor.fetchall()
        
        for row in rows:
            name, raw_spec_json, base_url, headers_json = row
            raw = json.loads(raw_spec_json)
            headers = json.loads(headers_json) if headers_json else {}
            
            self._cache[name] = SpecRecord(
                name=name,
                raw=raw,
                base_url=base_url,
                headers=headers,
                parser=None
            )
        
        conn.close()
        log.info(f"Loaded {len(self._cache)} specs from database")
    
    def upsert(self, rec: SpecRecord) -> None:
        """Insert or update a spec record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        raw_spec_json = json.dumps(rec.raw)
        headers_json = json.dumps(rec.headers)
        
        cursor.execute("""
            INSERT INTO openapi_specs (name, raw_spec, base_url, headers, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                raw_spec = excluded.raw_spec,
                base_url = excluded.base_url,
                headers = excluded.headers,
                updated_at = CURRENT_TIMESTAMP
        """, (rec.name, raw_spec_json, rec.base_url, headers_json))
        
        conn.commit()
        conn.close()
        
        # Update cache
        self._cache[rec.name] = rec
        log.info(f"Saved spec '{rec.name}' to database")
    
    def get(self, name: str) -> SpecRecord:
        """Get a spec record by name."""
        if name not in self._cache:
            raise KeyError(f"Spec '{name}' not found")
        return self._cache[name]
    
    def delete(self, name: str) -> bool:
        """Delete a spec record by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM openapi_specs WHERE name = ?", (name,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        if deleted:
            self._cache.pop(name, None)
            log.info(f"Deleted spec '{name}' from database")
        
        return deleted
    
    def list_names(self) -> list[str]:
        """List all spec names."""
        return sorted(self._cache.keys())
    
    def clear_all(self) -> int:
        """Clear all specs from database. Returns count of deleted specs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM openapi_specs")
        count = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM openapi_specs")
        conn.commit()
        conn.close()
        
        self._cache.clear()
        log.info(f"Cleared {count} specs from database")
        
        return count


store = DatabaseStore()


# ============================================================================
# Pydantic models for structured MCP tool I/O
# ============================================================================

class AddSpecInput(BaseModel):
    name: str = Field(description="Unique name to store/refer to this spec")
    source: str = Field(description="URL (http/https), file path (.json/.yaml), or raw JSON string")
    base_url: Optional[str] = Field(
        default=None,
        description="Override base URL; if not provided uses spec.servers[0].url when available"
    )
    headers: Optional[dict] = Field(default=None, description="Default headers for API calls")


class AddSpecResult(BaseModel):
    name: str
    title: Optional[str] = None
    version: Optional[str] = None
    server_urls: list[str] = []
    total_paths: int


class ApiOverview(BaseModel):
    operation_id: Optional[str] = Field(None, description="operationId if present")
    method: str
    path: str
    summary: Optional[str] = None
    tags: list[str] = []


class ListApisInput(BaseModel):
    name: str


class SearchApisInput(BaseModel):
    name: str
    keywords: list[str] = Field(description="Any words to match operation summary/tags/path")


class InvokeOperationInput(BaseModel):
    name: str
    operation_id: str
    path_params: dict = Field(default_factory=dict)
    query_params: dict = Field(default_factory=dict)
    body: Optional[dict] = None
    # Optional overrides:
    headers: Optional[dict] = None
    method_override: Optional[str] = None
    timeout_sec: float = 30.0
    base_url_override: Optional[str] = None


class InvokeOperationResult(BaseModel):
    status_code: int
    headers: dict
    body: Optional[dict] = None
    text: str
    url: str


class DeleteSpecInput(BaseModel):
    name: str = Field(description="Name of the spec to delete")


class DeleteSpecResult(BaseModel):
    name: str
    deleted: bool


class ListSpecsResult(BaseModel):
    specs: list[str]
    count: int


# ============================================================================
# Initialize FastMCP
# ============================================================================

mcp = FastMCP("Swagger MCP Server")


# ============================================================================
# Helper Functions
# ============================================================================

def load_source_text(source: str) -> str:
    """Load text from URL/path; otherwise treat source as raw JSON string."""
    if re.match(r"^https?://", source, flags=re.I):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        return resp.text
    
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            return f.read()
    
    # Assume raw JSON string
    return source


def parse_json_maybe_yaml(text: str) -> dict:
    """Parse JSON; attempt YAML if JSON fails (basic heuristic)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Very light YAML support for common cases (avoid adding pyyaml)
        # If you need full YAML, install PyYAML and replace this block.
        raise ValueError("Input is not valid JSON; please supply JSON or add YAML parsing.")


def get_base_url(spec: dict, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    servers = spec.get("servers") or []
    if servers and isinstance(servers, list) and isinstance(servers[0], dict):
        return servers[0].get("url")
    return None


def iter_operations(spec: dict) -> Iterable[ApiOverview]:
    """Yield ApiOverview from spec paths (for OpenAPI 3.x documents)."""
    paths = spec.get("paths") or {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete", "head", "options"):
                continue
            if not isinstance(operation, dict):
                continue
            yield ApiOverview(
                operation_id=operation.get("operationId"),
                method=method.upper(),
                path=path,
                summary=operation.get("summary") or operation.get("description"),
                tags=list(operation.get("tags") or [])
            )


def find_operation(spec: dict, operation_id: str) -> tuple[str, str, dict]:
    """Return (method, path, operation dict) by operationId."""
    for op in iter_operations(spec):
        if op.operation_id == operation_id:
            # Pull original operation dict to get requestBody details, etc.
            operation = spec["paths"][op.path][op.method.lower()]
            return op.method, op.path, operation
    raise KeyError(f"operationId '{operation_id}' not found")


def apply_path_params(path: str, params: dict) -> str:
    """Replace templated segments like /users/{id}.
    Leaves path unchanged if param missing."""
    for k, v in (params or {}).items():
        path = path.replace("{" + str(k) + "}", str(v))
    return path


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
def add_openapi_spec(args: AddSpecInput) -> AddSpecResult:
    """Add or update an OpenAPI spec. Accepts URL, file path, or raw JSON string.
    You can provide base_url and headers used for invocation.
    Specs are persisted to database and survive server restarts."""
    
    text = load_source_text(args.source)
    raw = parse_json_maybe_yaml(text)
    
    # Basic version detection
    is_oas3 = "openapi" in raw
    is_oas2 = "swagger" in raw and not is_oas3
    
    if is_oas2:
        log.warning(
            "We accept OAS2, but advise converting to OAS3 for best results. "
            "You can convert via converter.swagger.io or swagger2openapi CLI. "
            "(Still stored, but some features like requestBody may differ.)"
        )
    
    base_url = get_base_url(raw, args.base_url)
    headers = args.headers or {}
    
    rec = SpecRecord(
        name=args.name,
        raw=raw,
        base_url=base_url,
        headers=headers,
        parser=None
    )
    
    # Optional: attach openapi3-parser object if available & openapi 3.x
    if HAVE_OPENAPI_PARSER and is_oas3:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as tf:
                tf.write(json.dumps(raw))
                temp_path = tf.name
            rec.parser = parse_openapi(temp_path)  # type: ignore
            os.unlink(temp_path)
        except Exception:
            rec.parser = None  # non-critical
    
    store.upsert(rec)
    
    info = raw.get("info", {})
    server_urls = [s.get("url", "") for s in raw.get("servers", [])]
    
    return AddSpecResult(
        name=args.name,
        title=info.get("title"),
        version=info.get("version"),
        server_urls=server_urls,
        total_paths=len(raw.get("paths", {}))
    )


@mcp.tool()
def list_openapi_specs() -> ListSpecsResult:
    """List all stored OpenAPI spec names."""
    specs = store.list_names()
    return ListSpecsResult(specs=specs, count=len(specs))


@mcp.tool()
def delete_openapi_spec(args: DeleteSpecInput) -> DeleteSpecResult:
    """Delete an OpenAPI spec from the database."""
    try:
        deleted = store.delete(args.name)
        return DeleteSpecResult(name=args.name, deleted=deleted)
    except Exception as e:
        log.error(f"Error deleting spec '{args.name}': {e}")
        return DeleteSpecResult(name=args.name, deleted=False)


@mcp.tool()
def list_apis(args: ListApisInput) -> list[ApiOverview]:
    """List all operations (method/path/operationId/summary/tags) for the given spec name."""
    rec = store.get(args.name)
    return list(iter_operations(rec.raw))


@mcp.tool()
def search_apis(args: SearchApisInput) -> list[ApiOverview]:
    """Keyword search across summaries, tags, and path segments for a spec."""
    rec = store.get(args.name)
    kws = [k.lower() for k in args.keywords if k.strip()]
    
    out: list[ApiOverview] = []
    for op in iter_operations(rec.raw):
        hay = " ".join(filter(None, [
            " ".join(op.tags or []),
            op.summary or "",
            op.path
        ])).lower()
        
        if all(k in hay for k in kws):
            out.append(op)
    
    return out


@mcp.tool()
async def invoke_operation(
    args: InvokeOperationInput,
    ctx: Context[ServerSession, None]
) -> InvokeOperationResult:
    """Invoke an operation by operationId with provided params/body.
    Returns status code, headers, body/text, and final URL."""
    
    rec = store.get(args.name)
    method, path, operation = find_operation(rec.raw, args.operation_id)
    
    if args.method_override:
        method = args.method_override.upper()
    
    base_url = args.base_url_override or rec.base_url
    if not base_url:
        raise ValueError(
            "No base_url found. Provide base_url_override or ensure spec.servers[0].url is set."
        )
    
    # Build URL
    path_filled = apply_path_params(path, args.path_params)
    url = base_url.rstrip("/") + path_filled
    
    # Query
    if args.query_params:
        url += "?" + urlencode(args.query_params, doseq=True)
    
    # Headers (spec default + overrides)
    headers = dict(rec.headers or {})
    if args.headers:
        headers.update(args.headers)
    
    # Body
    data = None
    json_body = None
    if args.body is not None:
        # Prefer JSON body for OpenAPI
        json_body = args.body
    
    # Progress logging to MCP Client (optional)
    await ctx.info(f"Invoking {method} {url}")
    
    timeout = args.timeout_sec
    
    # Make request
    resp = requests.request(
        method=method,
        url=url,
        headers=headers,
        json=json_body,
        data=data,
        timeout=timeout
    )
    
    # Parse response
    try:
        body = resp.json()
    except Exception:
        body = None
    
    text = resp.text
    
    return InvokeOperationResult(
        status_code=resp.status_code,
        headers=dict(resp.headers),
        body=body,
        text=text,
        url=url
    )


# ============================================================================
# FastAPI HTTP App Setup
# ============================================================================

mcp_app = mcp.http_app(path="/", transport="streamable-http")


# Header Normalization Middleware
class MCPHeaderNormalizationMiddleware(BaseHTTPMiddleware):
    """Fix LiteLLM's duplicate MCP protocol version headers."""
    
    async def dispatch(self, request: Request, call_next):
        # Only process MCP endpoints
        if request.url.path.startswith("/"):
            protocol_version = request.headers.get("mcp-protocol-version", "")
            if protocol_version and "," in protocol_version:
                # Extract first version and remove duplicates
                first_version = protocol_version.split(",")[0].strip()
                
                # Update headers in request scope
                log.debug(f"Normalized mcp-protocol-version: {protocol_version} -> {first_version}")
                
                new_headers = []
                for name, value in request.headers.items():
                    if name.lower() == "mcp-protocol-version":
                        new_headers.append((name.encode(), first_version.encode()))
                    else:
                        new_headers.append((name.encode(), value.encode()))
                
                request.scope["headers"] = new_headers
        
        return await call_next(request)


# FastAPI Application
app = FastAPI(
    title="Swagger MCP Server",
    description="Model Context Protocol server for OpenAPI/Swagger spec management with persistent storage",
    version="1.0.0",
    lifespan=mcp.app.lifespan
)

# Add middleware
app.add_middleware(MCPHeaderNormalizationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"]
)


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    log.debug(f"{request.method} {request.url.path} {request.headers.get('user-agent', 'unknown')}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        log.debug(f"{request.method} {request.url.path} {response.status_code} ({duration:.3f}s)")
        return response
    except Exception as exc:
        duration = time.time() - start_time
        log.error(f"{request.method} {request.url.path} ERROR ({duration:.3f}s): {exc}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# Mount MCP Application
app.mount("/mcp", mcp_app)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    spec_count = len(store.list_names())
    return {
        "status": "healthy",
        "database": DB_PATH,
        "specs_loaded": spec_count
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting MCP Swagger Server with database at {DB_PATH}")
    log.info(f"Loaded {len(store.list_names())} specs from database")
    uvicorn.run(app, host="0.0.0.0", port=8000)
