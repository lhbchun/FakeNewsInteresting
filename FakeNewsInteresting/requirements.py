import subprocess
import sys
import nltk
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../requirements.txt"])
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')