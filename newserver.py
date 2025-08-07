# Save as https_server.py
import http.server
import ssl
import socketserver

PORT = 8443  # Common HTTPS port
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    # Create SSL context (modern approach)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('./server.crt', './server.key')
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    print(f"HTTPS Server running on port {PORT}")
    httpd.serve_forever()