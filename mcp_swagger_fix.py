“””
REST API MCP Server
A configurable MCP server that allows users to define API endpoints,
schemas, payloads, and make HTTP requests through tools.

Install: pip install fastmcp httpx pydantic
Run: python rest_api_mcp.py
“””

import json
import httpx
from typing import Optional, Any
from pydantic import BaseModel, Field
from fastmcp import FastMCP

# Initialize the MCP server

mcp = FastMCP(
“REST API Gateway”,
dependencies=[“httpx”, “pydantic”]
)

# In-memory storage for endpoint configurations

endpoint_configs: dict[str, dict] = {}

# ============================================================================

# PYDANTIC MODELS FOR TYPE SAFETY

# ============================================================================

class EndpointConfig(BaseModel):
“”“Configuration for an API endpoint”””
name: str = Field(…, description=“Unique name for this endpoint”)
base_url: str = Field(…, description=“Base URL (e.g., https://api.example.com)”)
path: str = Field(default=””, description=“API path (e.g., /users)”)
method: str = Field(default=“GET”, description=“HTTP method: GET, POST, PUT, PATCH, DELETE”)
headers: dict[str, str] = Field(default_factory=dict, description=“Default headers”)
query_params: dict[str, str] = Field(default_factory=dict, description=“Default query parameters”)
body_schema: dict[str, Any] = Field(default_factory=dict, description=“JSON schema for request body”)
auth_type: Optional[str] = Field(default=None, description=“Auth type: bearer, api_key, basic”)
auth_value: Optional[str] = Field(default=None, description=“Auth token/key value”)
timeout: float = Field(default=30.0, description=“Request timeout in seconds”)
description: str = Field(default=””, description=“Description of this endpoint”)

class AuthConfig(BaseModel):
“”“Authentication configuration”””
auth_type: str = Field(…, description=“Auth type: bearer, api_key, basic, custom_header”)
auth_value: str = Field(…, description=“Auth token/key/credentials”)
header_name: Optional[str] = Field(default=None, description=“Custom header name (for api_key/custom_header)”)

# ============================================================================

# ENDPOINT CONFIGURATION TOOLS

# ============================================================================

@mcp.tool()
def add_endpoint(
name: str,
base_url: str,
path: str = “”,
method: str = “GET”,
headers: Optional[dict] = None,
query_params: Optional[dict] = None,
body_schema: Optional[dict] = None,
auth_type: Optional[str] = None,
auth_value: Optional[str] = None,
timeout: float = 30.0,
description: str = “”
) -> dict:
“””
Add or update an API endpoint configuration.


Args:
    name: Unique identifier for this endpoint
    base_url: Base URL (e.g., https://api.example.com)
    path: API path (e.g., /v1/users)
    method: HTTP method (GET, POST, PUT, PATCH, DELETE)
    headers: Default headers as key-value pairs
    query_params: Default query parameters
    body_schema: JSON schema defining expected request body structure
    auth_type: Authentication type (bearer, api_key, basic, custom_header)
    auth_value: Authentication token or key
    timeout: Request timeout in seconds
    description: Human-readable description

Returns:
    Confirmation with endpoint details
"""
config = {
    "name": name,
    "base_url": base_url.rstrip("/"),
    "path": path if path.startswith("/") or not path else f"/{path}",
    "method": method.upper(),
    "headers": headers or {},
    "query_params": query_params or {},
    "body_schema": body_schema or {},
    "auth_type": auth_type,
    "auth_value": auth_value,
    "timeout": timeout,
    "description": description
}

endpoint_configs[name] = config

return {
    "status": "success",
    "message": f"Endpoint '{name}' configured successfully",
    "endpoint": {k: v for k, v in config.items() if k != "auth_value"}
}


@mcp.tool()
def list_endpoints() -> dict:
“””
List all configured API endpoints.


Returns:
    Dictionary of all endpoint configurations (auth values hidden)
"""
if not endpoint_configs:
    return {"status": "empty", "message": "No endpoints configured", "endpoints": []}

safe_configs = []
for name, config in endpoint_configs.items():
    safe_config = {k: v for k, v in config.items() if k != "auth_value"}
    safe_config["has_auth"] = config.get("auth_value") is not None
    safe_configs.append(safe_config)

return {
    "status": "success",
    "count": len(safe_configs),
    "endpoints": safe_configs
}


@mcp.tool()
def get_endpoint(name: str) -> dict:
“””
Get details of a specific endpoint configuration.


Args:
    name: Name of the endpoint to retrieve

Returns:
    Endpoint configuration details
"""
if name not in endpoint_configs:
    return {"status": "error", "message": f"Endpoint '{name}' not found"}

config = endpoint_configs[name]
safe_config = {k: v for k, v in config.items() if k != "auth_value"}
safe_config["has_auth"] = config.get("auth_value") is not None

return {"status": "success", "endpoint": safe_config}


@mcp.tool()
def remove_endpoint(name: str) -> dict:
“””
Remove an endpoint configuration.


Args:
    name: Name of the endpoint to remove

Returns:
    Confirmation of removal
"""
if name not in endpoint_configs:
    return {"status": "error", "message": f"Endpoint '{name}' not found"}

del endpoint_configs[name]
return {"status": "success", "message": f"Endpoint '{name}' removed"}


@mcp.tool()
def update_endpoint_auth(
name: str,
auth_type: str,
auth_value: str,
header_name: Optional[str] = None
) -> dict:
“””
Update authentication for an existing endpoint.


Args:
    name: Name of the endpoint
    auth_type: Auth type (bearer, api_key, basic, custom_header)
    auth_value: Auth token/key/credentials
    header_name: Custom header name (required for api_key/custom_header)

Returns:
    Confirmation of update
"""
if name not in endpoint_configs:
    return {"status": "error", "message": f"Endpoint '{name}' not found"}

endpoint_configs[name]["auth_type"] = auth_type
endpoint_configs[name]["auth_value"] = auth_value

if header_name and auth_type in ["api_key", "custom_header"]:
    endpoint_configs[name]["headers"][header_name] = auth_value

return {"status": "success", "message": f"Auth updated for '{name}'"}


# ============================================================================

# API CALL TOOLS

# ============================================================================

def _build_headers(config: dict, extra_headers: Optional[dict] = None) -> dict:
“”“Build request headers including authentication”””
headers = {“Content-Type”: “application/json”, **config.get(“headers”, {})}


if extra_headers:
    headers.update(extra_headers)

auth_type = config.get("auth_type")
auth_value = config.get("auth_value")

if auth_type and auth_value:
    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {auth_value}"
    elif auth_type == "basic":
        import base64
        encoded = base64.b64encode(auth_value.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"
    elif auth_type == "api_key":
        headers["X-API-Key"] = auth_value

return headers


@mcp.tool()
async def call_endpoint(
name: str,
path_override: Optional[str] = None,
query_params: Optional[dict] = None,
body: Optional[dict] = None,
headers: Optional[dict] = None
) -> dict:
“””
Make an API call using a configured endpoint.


Args:
    name: Name of the configured endpoint to use
    path_override: Override the configured path (optional)
    query_params: Additional query parameters (merged with defaults)
    body: Request body (for POST/PUT/PATCH)
    headers: Additional headers (merged with defaults)

Returns:
    API response with status, headers, and body
"""
if name not in endpoint_configs:
    return {"status": "error", "message": f"Endpoint '{name}' not found"}

config = endpoint_configs[name]

# Build URL
path = path_override if path_override else config.get("path", "")
url = f"{config['base_url']}{path}"

# Merge query params
params = {**config.get("query_params", {}), **(query_params or {})}

# Build headers
req_headers = _build_headers(config, headers)

method = config.get("method", "GET").upper()
timeout = config.get("timeout", 30.0)

try:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method=method,
            url=url,
            params=params if params else None,
            json=body if body and method in ["POST", "PUT", "PATCH"] else None,
            headers=req_headers
        )
        
        # Try to parse JSON response
        try:
            response_body = response.json()
        except:
            response_body = response.text
        
        return {
            "status": "success",
            "http_status": response.status_code,
            "url": str(response.url),
            "method": method,
            "response_headers": dict(response.headers),
            "body": response_body
        }
        
except httpx.TimeoutException:
    return {"status": "error", "message": "Request timed out"}
except httpx.RequestError as e:
    return {"status": "error", "message": f"Request failed: {str(e)}"}


@mcp.tool()
async def quick_request(
url: str,
method: str = “GET”,
headers: Optional[dict] = None,
query_params: Optional[dict] = None,
body: Optional[dict] = None,
auth_type: Optional[str] = None,
auth_value: Optional[str] = None,
timeout: float = 30.0
) -> dict:
“””
Make a quick one-off API request without saving configuration.


Args:
    url: Full URL to call
    method: HTTP method (GET, POST, PUT, PATCH, DELETE)
    headers: Request headers
    query_params: Query parameters
    body: Request body (for POST/PUT/PATCH)
    auth_type: Auth type (bearer, api_key, basic)
    auth_value: Auth token/key
    timeout: Request timeout in seconds

Returns:
    API response with status, headers, and body
"""
req_headers = {"Content-Type": "application/json", **(headers or {})}

# Add authentication
if auth_type and auth_value:
    if auth_type == "bearer":
        req_headers["Authorization"] = f"Bearer {auth_value}"
    elif auth_type == "basic":
        import base64
        encoded = base64.b64encode(auth_value.encode()).decode()
        req_headers["Authorization"] = f"Basic {encoded}"
    elif auth_type == "api_key":
        req_headers["X-API-Key"] = auth_value

method = method.upper()

try:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method=method,
            url=url,
            params=query_params if query_params else None,
            json=body if body and method in ["POST", "PUT", "PATCH"] else None,
            headers=req_headers
        )
        
        try:
            response_body = response.json()
        except:
            response_body = response.text
        
        return {
            "status": "success",
            "http_status": response.status_code,
            "url": str(response.url),
            "method": method,
            "response_headers": dict(response.headers),
            "body": response_body
        }
        
except httpx.TimeoutException:
    return {"status": "error", "message": "Request timed out"}
except httpx.RequestError as e:
    return {"status": "error", "message": f"Request failed: {str(e)}"}


# ============================================================================

# SCHEMA TOOLS

# ============================================================================

@mcp.tool()
def set_body_schema(name: str, schema: dict) -> dict:
“””
Set or update the request body schema for an endpoint.


Args:
    name: Name of the endpoint
    schema: JSON Schema defining the expected request body structure

Returns:
    Confirmation with the schema

Example schema:
    {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["username", "email"]
    }
"""
if name not in endpoint_configs:
    return {"status": "error", "message": f"Endpoint '{name}' not found"}

endpoint_configs[name]["body_schema"] = schema
return {
    "status": "success",
    "message": f"Schema updated for '{name}'",
    "schema": schema
}


@mcp.tool()
def get_body_schema(name: str) -> dict:
“””
Get the request body schema for an endpoint.


Args:
    name: Name of the endpoint

Returns:
    The JSON schema for the request body
"""
if name not in endpoint_configs:
    return {"status": "error", "message": f"Endpoint '{name}' not found"}

schema = endpoint_configs[name].get("body_schema", {})
return {
    "status": "success",
    "endpoint": name,
    "schema": schema if schema else "No schema defined"
}


@mcp.tool()
def validate_body(name: str, body: dict) -> dict:
“””
Validate a request body against the endpoint’s schema.


Args:
    name: Name of the endpoint
    body: Request body to validate

Returns:
    Validation result with any errors
"""
if name not in endpoint_configs:
    return {"status": "error", "message": f"Endpoint '{name}' not found"}

schema = endpoint_configs[name].get("body_schema", {})

if not schema:
    return {"status": "warning", "message": "No schema defined, body accepted"}

errors = []

# Basic validation based on schema
if schema.get("type") == "object":
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Check required fields
    for field in required:
        if field not in body:
            errors.append(f"Missing required field: {field}")
    
    # Check field types
    for field, value in body.items():
        if field in properties:
            expected_type = properties[field].get("type")
            if expected_type:
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "array": list,
                    "object": dict
                }
                expected = type_map.get(expected_type)
                if expected and not isinstance(value, expected):
                    errors.append(f"Field '{field}' should be {expected_type}")

if errors:
    return {"status": "invalid", "errors": errors}

return {"status": "valid", "message": "Body matches schema"}


# ============================================================================

# BULK/BATCH OPERATIONS

# ============================================================================

@mcp.tool()
async def batch_requests(requests: list[dict]) -> dict:
“””
Execute multiple API requests in sequence.


Args:
    requests: List of request configs, each containing:
        - endpoint: Name of configured endpoint OR
        - url: Direct URL for quick request
        - path_override: Optional path override
        - query_params: Optional query params
        - body: Optional request body
        - headers: Optional headers

Returns:
    Results of all requests
"""
results = []

for i, req in enumerate(requests):
    if "endpoint" in req:
        result = await call_endpoint(
            name=req["endpoint"],
            path_override=req.get("path_override"),
            query_params=req.get("query_params"),
            body=req.get("body"),
            headers=req.get("headers")
        )
    elif "url" in req:
        result = await quick_request(
            url=req["url"],
            method=req.get("method", "GET"),
            headers=req.get("headers"),
            query_params=req.get("query_params"),
            body=req.get("body")
        )
    else:
        result = {"status": "error", "message": "Missing 'endpoint' or 'url'"}
    
    results.append({"index": i, "request": req, "result": result})

return {
    "status": "success",
    "total": len(results),
    "results": results
}


# ============================================================================

# EXPORT/IMPORT CONFIGURATION

# ============================================================================

@mcp.tool()
def export_config() -> dict:
“””
Export all endpoint configurations (without auth values).


Returns:
    JSON-serializable configuration that can be saved and imported
"""
export_data = {}
for name, config in endpoint_configs.items():
    safe_config = {k: v for k, v in config.items() if k != "auth_value"}
    export_data[name] = safe_config

return {
    "status": "success",
    "config": export_data
}


@mcp.tool()
def import_config(config: dict) -> dict:
“””
Import endpoint configurations.


Args:
    config: Dictionary of endpoint configurations to import

Returns:
    Import results
"""
imported = []
errors = []

for name, endpoint_config in config.items():
    try:
        endpoint_configs[name] = {
            "name": name,
            "base_url": endpoint_config.get("base_url", ""),
            "path": endpoint_config.get("path", ""),
            "method": endpoint_config.get("method", "GET"),
            "headers": endpoint_config.get("headers", {}),
            "query_params": endpoint_config.get("query_params", {}),
            "body_schema": endpoint_config.get("body_schema", {}),
            "auth_type": endpoint_config.get("auth_type"),
            "auth_value": endpoint_config.get("auth_value"),
            "timeout": endpoint_config.get("timeout", 30.0),
            "description": endpoint_config.get("description", "")
        }
        imported.append(name)
    except Exception as e:
        errors.append({"name": name, "error": str(e)})

return {
    "status": "success" if not errors else "partial",
    "imported": imported,
    "errors": errors if errors else None
}


# ============================================================================

# RESOURCES - READ-ONLY DATA

# ============================================================================

@mcp.resource(“endpoints://list”)
def resource_list_endpoints() -> str:
“”“Get a formatted list of all configured endpoints”””
if not endpoint_configs:
return “No endpoints configured”


lines = ["# Configured API Endpoints\n"]
for name, config in endpoint_configs.items():
    lines.append(f"## {name}")
    lines.append(f"- URL: {config['base_url']}{config.get('path', '')}")
    lines.append(f"- Method: {config.get('method', 'GET')}")
    lines.append(f"- Auth: {'Configured' if config.get('auth_value') else 'None'}")
    if config.get('description'):
        lines.append(f"- Description: {config['description']}")
    lines.append("")

return "\n".join(lines)


@mcp.resource(“endpoints://{name}/details”)
def resource_endpoint_details(name: str) -> str:
“”“Get detailed information about a specific endpoint”””
if name not in endpoint_configs:
return f”Endpoint ‘{name}’ not found”


config = endpoint_configs[name]
return json.dumps({k: v for k, v in config.items() if k != "auth_value"}, indent=2)


# ============================================================================

# RUN THE SERVER

# ============================================================================

if *name* == “*main*”:
mcp.run()

















{
  "openapi": "3.0.0",
  "info": {
    "title": "VirusTotal API v3.0",
    "description": "API for scanning files, URLs, domains, and IPs with extended features and metadata.",
    "version": "3.0"
  },
  "servers": [
    {
      "url": "https://www.virustotal.com/api/v3",
      "description": "Main VirusTotal API server"
    }
  ],
  "components": {
    "securitySchemes": {
      "ApiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "x-apikey",
        "description": "Your API key goes in the x-apikey header for authentication."
      }
    },
    "schemas": {
      "FileReport": {
        "type": "object",
        "properties": {
          "data": {
            "type": "object",
            "properties": {
              "attributes": {
                "type": "object",
                "properties": {
                  "last_analysis_stats": {
                    "type": "object",
                    "properties": {
                      "harmless": {
                        "type": "integer"
                      },
                      "malicious": {
                        "type": "integer"
                      },
                      "suspicious": {
                        "type": "integer"
                      },
                      "undetected": {
                        "type": "integer"
                      }
                    }
                  },
                  "last_analysis_results": {
                    "type": "object",
                    "additionalProperties": {
                      "type": "object",
                      "properties": {
                        "category": {
                          "type": "string"
                        },
                        "result": {
                          "type": "string"
                        }
                      }
                    }
                  },
                  "sha256": {
                    "type": "string"
                  },
                  "md5": {
                    "type": "string"
                  },
                  "sha1": {
                    "type": "string"
                  },
                  "size": {
                    "type": "integer"
                  },
                  "tags": {
                    "type": "array",
                    "items": {
                      "type": "string"
                    }
                  }
                }
              }
            }
          },
          "links": {
            "type": "object",
            "properties": {
              "self": {
                "type": "string"
              }
            }
          }
        }
      }
    }
  },
  "paths": {
    "/files/{file_id}": {
      "get": {
        "summary": "Retrieve file scan report by file ID (SHA256)",
        "parameters": [
          {
            "name": "file_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            },
            "description": "SHA256 hash of the file."
          }
        ],
        "responses": {
          "200": {
            "description": "Successful response with file report.",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/FileReport"
                }
              }
            }
          },
          "400": {
            "description": "Bad request."
          }
        },
        "security": [
          {
            "ApiKeyAuth": []
          }
        ]
      }
    },
    "/urls/{url_id}": {
      "get": {
        "summary": "Retrieve URL scan report by URL ID (SHA256)",
        "parameters": [
          {
            "name": "url_id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            },
            "description": "Encoded URL identifier (SHA256)."
          }
        ],
        "responses": {
          "200": {
            "description": "Successful response with URL report.",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/FileReport"
                }
              }
            }
          },
          "400": {
            "description": "Bad request."
          }
        },
        "security": [
          {
            "ApiKeyAuth": []
          }
        ]
      }
    },
    "/domains/{domain_name}": {
      "get": {
        "summary": "Retrieve domain report by domain name.",
        "parameters": [
          {
            "name": "domain_name",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            },
            "description": "Domain name to retrieve the report for."
          }
        ],
        "responses": {
          "200": {
            "description": "Successful response with domain report.",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/FileReport"
                }
              }
            }
          },
          "400": {
            "description": "Bad request."
          }
        },
        "security": [
          {
            "ApiKeyAuth": []
          }
        ]
      }
    },
    "/ip_addresses/{ip_address}": {
      "get": {
        "summary": "Retrieve IP address report by IP address.",
        "parameters": [
          {
            "name": "ip_address",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string"
            },
            "description": "IP address to retrieve the report for."
          }
        ],
        "responses": {
          "200": {
            "description": "Successful response with IP address report.",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/FileReport"
                }
              }
            }
          },
          "400": {
            "description": "Bad request."
          }
        },
        "security": [
          {
            "ApiKeyAuth": []
          }
        ]











        import pandas as pd
import re
from collections import defaultdict
from typing import List, Dict, Optional

# ── Sample Data ────────────────────────────────────────────────────────────────
data = {
    "id":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "qid": ["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"],
    "question": [
        "What is machine learning?",
        "How does deep learning differ from machine learning?",
        "What is supervised learning?",
        "Explain unsupervised learning with examples.",
        "What is natural language processing?",
        "How do neural networks work?",
        "What is reinforcement learning?",
        "How is deep learning used in computer vision?",
        "What are transformers in NLP?",
        "How does BERT work in natural language processing?",
    ]
}
df = pd.DataFrame(data)


# ── PageIndex ──────────────────────────────────────────────────────────────────
class PageIndex:
    """
    Vectorless full-text index over a DataFrame.

    Strategy:
      1. Split the DataFrame into fixed-size pages.
      2. For each page, build an inverted index: token → set of row positions.
      3. At query time, tokenize the query, intersect (AND) or union (OR)
         candidate row sets across pages, then return matching rows.
    """

    def __init__(self, df: pd.DataFrame, text_col: str, page_size: int = 3):
        self.df        = df.reset_index(drop=True)
        self.text_col  = text_col
        self.page_size = page_size
        self.pages: List[Dict] = []   # list of {page_id, row_range, index}
        self._build()

    # ── tokeniser ─────────────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    # ── build inverted index per page ─────────────────────────────────────────
    def _build(self):
        n = len(self.df)
        for page_id, start in enumerate(range(0, n, self.page_size)):
            end      = min(start + self.page_size, n)
            page_df  = self.df.iloc[start:end]
            inv_idx: Dict[str, set] = defaultdict(set)

            for local_pos, (abs_idx, row) in enumerate(page_df.iterrows()):
                for token in self._tokenize(str(row[self.text_col])):
                    inv_idx[token].add(abs_idx)   # store absolute df index

            self.pages.append({
                "page_id"  : page_id,
                "row_range": (start, end - 1),
                "index"    : inv_idx,
            })

        print(f"[PageIndex] Built {len(self.pages)} pages "
              f"(page_size={self.page_size}) over {n} rows.")

    # ── query ─────────────────────────────────────────────────────────────────
    def query(self, query_str: str, mode: str = "AND") -> pd.DataFrame:
        """
        Parameters
        ----------
        query_str : str   – free-text query
        mode      : str   – "AND" (all terms must match) | "OR" (any term matches)

        Returns
        -------
        pd.DataFrame of matching rows.
        """
        tokens = self._tokenize(query_str)
        if not tokens:
            return pd.DataFrame(columns=self.df.columns)

        print(f"\n[Query] '{query_str}'  mode={mode}  tokens={tokens}")

        candidate_rows: Optional[set] = None

        for page in self.pages:
            inv_idx = page["index"]

            # collect matching row-sets for each token in this page
            token_sets = [inv_idx.get(tok, set()) for tok in tokens]

            if mode == "AND":
                page_matches = token_sets[0].copy()
                for s in token_sets[1:]:
                    page_matches &= s
            else:  # OR
                page_matches = set()
                for s in token_sets:
                    page_matches |= s

            if candidate_rows is None:
                candidate_rows = page_matches
            else:
                candidate_rows |= page_matches   # union across pages

        if not candidate_rows:
            print("[Query] No matches found.")
            return pd.DataFrame(columns=self.df.columns)

        result = self.df.loc[sorted(candidate_rows)]
        print(f"[Query] {len(result)} row(s) found.")
        return result


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    idx = PageIndex(df, text_col="question", page_size=3)

    print("\n" + "="*60)
    print("SEARCH 1 — AND: 'machine learning'")
    print("="*60)
    print(idx.query("machine learning", mode="AND").to_string(index=False))

    print("\n" + "="*60)
    print("SEARCH 2 — OR: 'deep learning'")
    print("="*60)
    print(idx.query("deep learning", mode="OR").to_string(index=False))

    print("\n" + "="*60)
    print("SEARCH 3 — AND: 'natural language processing'")
    print("="*60)
    print(idx.query("natural language processing", mode="AND").to_string(index=False))

    print("\n" + "="*60)
    print("SEARCH 4 — AND: 'neural networks'")
    print("="*60)
    print(idx.query("neural networks", mode="AND").to_string(index=False))
      }
    }
  }
}
