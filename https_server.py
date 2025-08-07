import http.server
import ssl
import socketserver
import mimetypes

# Add proper MIME type for .mjs files
mimetypes.add_type('application/javascript', '.mjs')

class WebGPUHTTPSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Essential headers for WebGPU and cross-origin isolation
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        
        # CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

PORT = 8443
with socketserver.TCPServer(('', PORT), WebGPUHTTPSRequestHandler) as httpd:
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    print(f'Serving at https://0.0.0.0:{PORT}/')
    print('You will need to accept the self-signed certificate warning in your browser')
    httpd.serve_forever()
