#!/usr/bin/env python3
"""
Food Price Predictor - Web Application Startup Script
This script starts the web server for the food price prediction application.
"""

import os
import webbrowser
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

def start_server():
    """Start the HTTP server"""
    port = 8000
    server_address = ('', port)
    
    class MyHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory='.', **kwargs)
    
    httpd = HTTPServer(server_address, MyHandler)
    print(f"Server running at http://localhost:{port}")
    print("Food Price Predictor is ready!")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
        httpd.shutdown()

def main():
    """Main function"""
    print("Starting Food Price Predictor Web Application")
    print("=" * 60)
    
    # Check if model files exist
    if not os.path.exists('models/models_index.json'):
        print("Model files not found!")
        print("Please run: python create_models_from_pkl.py")
        return
    
    print("Model files found")
    print("Starting web server...")
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(1)
    
    # Open browser automatically
    try:
        webbrowser.open('http://localhost:8000')
        print("Opened application in your default browser")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please visit: http://localhost:8000")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main() 