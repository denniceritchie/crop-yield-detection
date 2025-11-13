import os

# Secret key for session management
SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
    'sqlite:///' + os.path.join(basedir, 'instance', 'database.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False