import logging
from flask import jsonify
from werkzeug.exceptions import HTTPException
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def register_error_handlers(app):

    @app.errorhandler(ValidationError)
    def handle_pydantic_error(e):
        logger.warning(f"Pydantic validation error: {e.errors()}")
        return jsonify({
            "success": False,
            "error": {"code": "VALIDATION_ERROR", "message": str(e), "details": e.errors()}
        }), 400

    @app.errorhandler(HTTPException)
    def handle_http_exception(e):
        return jsonify({
            "success": False,
            "error": {"code": f"HTTP_{e.code}", "message": e.description}
        }), e.code

    @app.errorhandler(Exception)
    def handle_generic_exception(e):
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": {"code": "INTERNAL_SERVER_ERROR", "message": str(e)}
        }), 500
