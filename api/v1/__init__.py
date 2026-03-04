from flask import Blueprint

scan_bp = Blueprint('scan_bp', __name__)

from . import scan_routes
