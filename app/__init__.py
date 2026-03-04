from flask import Flask
from flask_cors import CORS
from app.api.v1.scan_routes import scan_bp
from app.api.middleware.error_handler import register_error_handlers
from app.infrastructure.model_loader import model_manager
from app.config import Config
import logging

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')

    CORS(app)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    app.register_blueprint(scan_bp, url_prefix='/api/v1/scans')

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load ML models at startup
    model_manager.load_models(Config)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return {"success": True, "data": {"status": "up", "version": "1.0.0"}, "error": None}, 200

    return app
