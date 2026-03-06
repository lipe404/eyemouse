import sys
import os

# Add the eye_mouse directory to sys.path so that 'import config' works inside the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'eye_mouse')))
