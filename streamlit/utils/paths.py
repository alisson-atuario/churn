import sys
from pathlib import Path
import joblib

# Define raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent.parent
print(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
SRC_DIR = PROJECT_ROOT / 'src'


# Adiciona src ao path para imports
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
    sys.path.append(str(SRC_DIR / 'models'))
