#!/usr/bin/env python3
"""
HTTP server with Cross-Origin-Embedder-Policy and Cross-Origin-Opener-Policy headers
for enabling crossOriginIsolated mode (required for WebGPU and SharedArrayBuffer)
"""

import http.server
import socketserver
import sys
from urllib.parse import urlparse

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add headers required for cross-origin isolation
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        # Additional headers that might be useful
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        super().end_headers()

    def log_message(self, format, *args):
        # Custom log format to show the isolation headers are being sent
        super().log_message(format, *args)

if __name__ == "__main__":
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}/")
        print("Cross-Origin-Isolation enabled (required for WebGPU)")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
