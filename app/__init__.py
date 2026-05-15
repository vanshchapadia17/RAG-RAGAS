from flask import Flask
from src.config import FLASK_SECRET_KEY

def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )
    app.secret_key = FLASK_SECRET_KEY

    from app.routes.chat import chat_bp
    from app.routes.eval import eval_bp
    from app.routes.experiments import experiments_bp
    from app.routes.documents import documents_bp

    app.register_blueprint(chat_bp)
    app.register_blueprint(eval_bp)
    app.register_blueprint(experiments_bp)
    app.register_blueprint(documents_bp)

    return app