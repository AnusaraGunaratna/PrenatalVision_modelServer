from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from app.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')

    CORS(app)

    # Late imports to prevent Circular Dependency/Incomplete package loading in Gunicorn
    from app.api.v1.scan_routes import scan_bp
    from app.api.middleware.error_handler import register_error_handlers
    
    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    app.register_blueprint(scan_bp, url_prefix='/api/v1/scans')

    # Setup basic logging

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



    # API Key protection(Springboot access control)
    @app.before_request
    def verify_api_key():
        if request.path == '/api/health':
            return None
        api_key = request.headers.get('X-API-Key')
        if api_key != Config.API_KEY:
            return jsonify({"success": False, "error": {"code": "UNAUTHORIZED", "message": "Invalid or missing API key"}}), 401

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return {"success": True, "data": {"status": "up", "version": "1.0.0"}, "error": None}, 200

    return app
