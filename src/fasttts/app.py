"""FastAPI application factory and entry point."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .engines import engine_manager
from .routes import tts_router, voices_router, websocket_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Configure logging
    if settings.debug_logging:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Initialize engines
    engine_manager.init_engines()

    app = FastAPI(title="FastTTS API", version="0.1.0")

    # Mount static files
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Content Security Policy
    csp = {
        "default-src": "'self'",
        "script-src": "'self'",
        "style-src": "'self' 'unsafe-inline'",
        "img-src": "'self' data:",
        "font-src": "'self' data:",
        "media-src": "'self' blob:",
    }
    csp_string = "; ".join(f"{key} {value}" for key, value in csp.items())

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = csp_string
        return response

    # Include routers
    app.include_router(tts_router)
    app.include_router(voices_router)
    app.include_router(websocket_router)

    # Static routes
    @app.get("/favicon.ico")
    async def favicon():
        favicon_path = static_dir / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(str(favicon_path))
        return FileResponse("static/favicon.ico")

    @app.get("/")
    def root_page():
        """Serve the main HTML page."""
        engines_options = "".join(
            [
                f'<option value="{engine}">{engine.title()}</option>'
                for engine in engine_manager.engines.keys()
            ]
        )
        content = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Text-To-Speech</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #f0f0f0;
                        margin: 0;
                        padding: 0;
                    }}
                    h2 {{
                        color: #333;
                        text-align: center;
                    }}
                    #container {{
                        width: 80%;
                        margin: 50px auto;
                        background-color: #fff;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}
                    label {{
                        font-weight: bold;
                    }}
                    select, textarea {{
                        width: 100%;
                        padding: 10px;
                        margin: 10px 0;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        box-sizing: border-box;
                        font-size: 16px;
                    }}
                    button {{
                        display: block;
                        width: 100%;
                        padding: 15px;
                        background-color: #007bff;
                        border: none;
                        border-radius: 5px;
                        color: #fff;
                        font-size: 16px;
                        cursor: pointer;
                        transition: background-color 0.3s;
                    }}
                    button:hover {{
                        background-color: #0056b3;
                    }}
                    audio {{
                        width: 80%;
                        margin: 10px auto;
                        display: block;
                    }}
                    .mode-selector {{
                        display: flex;
                        justify-content: center;
                        gap: 10px;
                        margin: 20px 0;
                    }}
                    .mode-selector button {{
                        width: auto;
                        padding: 10px 20px;
                    }}
                    .mode-selector button.active {{
                        background-color: #28a745;
                    }}
                    .status {{
                        text-align: center;
                        margin: 10px 0;
                        color: #666;
                        font-size: 14px;
                    }}
                </style>
            </head>
            <body>
                <div id="container">
                    <h2>Text to Speech</h2>
                    <div class="mode-selector">
                        <button id="httpMode" class="active">HTTP Mode</button>
                        <button id="wsMode">WebSocket Mode</button>
                    </div>
                    <div class="status" id="status">Mode: HTTP</div>
                    <label for="engine">Select Engine:</label>
                    <select id="engine">
                        {engines_options}
                    </select>
                    <label for="voice">Select Voice:</label>
                    <select id="voice">
                        <!-- Options will be dynamically populated by JavaScript -->
                    </select>
                    <textarea id="text" rows="4" cols="50" placeholder="Enter text here..."></textarea>
                    <button id="speakButton">Speak</button>
                    <audio id="audio" controls preload="auto" playsinline webkit-playsinline></audio>
                </div>
                <script src="/static/tts.js"></script>
            </body>
        </html>
        """
        return HTMLResponse(content=content)

    return app


# Create the app instance
app = create_app()


def main():
    """Entry point for running the server directly."""
    import uvicorn
    print("Server ready")
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
