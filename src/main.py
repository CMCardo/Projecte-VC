import subprocess
import sys
import os

def main():
    # Busquem la ruta absoluta d'app.py per evitar problemes de directoris
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")
    
    print("Iniciant Mountain Sky Remover")
    
    # Aixequem Streamlit com a subprocés
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    main()