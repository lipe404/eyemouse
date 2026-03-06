import PyInstaller.__main__
import os
import shutil


def build():
    # Caminhos
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(base_dir, "eye_mouse", "main.py")
    model_file = os.path.join(base_dir, "eye_mouse", "face_landmarker.task")
    dist_dir = os.path.join(base_dir, "dist")

    # Verificar existência do modelo
    if not os.path.exists(model_file):
        print(f"ERRO: Arquivo de modelo não encontrado: {model_file}")
        print(
            "Certifique-se de ter executado o programa pelo menos uma vez ou baixado o modelo."
        )
        return

    print("--- Iniciando Build do EyeMouse ---")

    # Argumentos do PyInstaller
    args = [
        main_script,
        "--name=EyeMouse",
        "--onefile",  # Arquivo único .exe
        "--windowed",  # Sem janela de console (GUI app)
        "--clean",  # Limpar cache antes
        "--noconfirm",  # Não perguntar para sobrescrever
        # Adicionar o diretório do código fonte ao path de busca
        f'--paths={os.path.join(base_dir, "eye_mouse")}',
        # Incluir o modelo na raiz do executável (.)
        # No Windows o separador é ;
        f"--add-data={model_file};.",
        # Coletar tudo do mediapipe para evitar erros de importação dinâmica
        "--collect-all=mediapipe",
        "--hidden-import=keyboard",
        # Opcional: Adicionar ícone se tiver (ex: --icon=icon.ico)
    ]

    try:
        PyInstaller.__main__.run(args)
        print("\n--- Build Concluído com Sucesso! ---")
        print(f"O executável está em: {os.path.join(dist_dir, 'EyeMouse.exe')}")
    except Exception as e:
        print(f"\nERRO durante o build: {e}")


if __name__ == "__main__":
    build()
