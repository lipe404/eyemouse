"""
build_exe.py — Script de compilação e empacotamento do EyeMouse para Windows.

Milestone 7:
  - Empacotamento reproduzível via PyInstaller.
  - Inclusão estrita do modelo MediaPipe (face_landmarker.task) em múltiplos pontos de busca.
  - Suporte a --onedir (rápida inicialização) e --onefile (arquivo único).
  - Coleta abrangente de dependências dinâmicas (MediaPipe, OpenCV, PIL).
  - Orientações de distribuição, assinatura digital Authenticode e Microsoft SmartScreen.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import PyInstaller.__main__


def build(mode: str = "onedir") -> bool:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(base_dir, "eye_mouse", "main.py")
    model_file = os.path.join(base_dir, "eye_mouse", "face_landmarker.task")
    dist_dir = os.path.join(base_dir, "dist")

    # 1. Validação de pré-requisitos
    if not os.path.exists(model_file):
        root_model = os.path.join(base_dir, "face_landmarker.task")
        if os.path.exists(root_model):
            shutil.copyfile(root_model, model_file)
        else:
            print(f"ERRO: Arquivo de modelo não encontrado em: {model_file}")
            print("Certifique-se de que o arquivo 'face_landmarker.task' existe no repositório.")
            return False

    print(f"--- Iniciando Build do EyeMouse (Modo: {mode}) ---")

    # 2. Argumentos do PyInstaller
    args = [
        main_script,
        "--name=EyeMouse",
        f"--{mode}",           # --onedir ou --onefile
        "--windowed",          # Sem janela preta de console
        "--clean",             # Limpa cache temporário anterior
        "--noconfirm",         # Sobrescreve sem perguntar
        f'--paths={os.path.join(base_dir, "eye_mouse")}',
        # Garante inclusão do modelo na raiz e na pasta do pacote
        f"--add-data={model_file};.",
        f"--add-data={model_file};eye_mouse",
        # Coleta de dependências dinâmicas
        "--collect-all=mediapipe",
        "--hidden-import=keyboard",
        "--hidden-import=cv2",
        "--hidden-import=PIL",
        "--hidden-import=numpy",
    ]

    try:
        PyInstaller.__main__.run(args)
        target_path = (
            os.path.join(dist_dir, "EyeMouse", "EyeMouse.exe")
            if mode == "onedir"
            else os.path.join(dist_dir, "EyeMouse.exe")
        )
        print("\n--- Build Concluído com Sucesso! ---")
        print(f"Executável gerado em: {target_path}")

        # 3. Verificação pós-build de integridade de arquivos
        if mode == "onedir":
            candidates = [
                os.path.join(dist_dir, "EyeMouse", "_internal", "face_landmarker.task"),
                os.path.join(dist_dir, "EyeMouse", "_internal", "eye_mouse", "face_landmarker.task"),
                os.path.join(dist_dir, "EyeMouse", "face_landmarker.task"),
                os.path.join(dist_dir, "EyeMouse", "eye_mouse", "face_landmarker.task"),
            ]
            if any(os.path.exists(c) for c in candidates):
                print("[✓] Modelo neural face_landmarker.task incluído com sucesso no pacote executável!")
            else:
                print("[!] Atenção: Modelo neural pode não ter sido copiado para a pasta do bundle.")

        print("\n--- Diretrizes de Distribuição e Segurança do Windows ---")
        print("1. Reputação do SmartScreen:")
        print("   Binários recém-compilados podem gerar aviso de 'Aplicativo não reconhecido' no Windows.")
        print("   NUNCA instrua o usuário final a desativar o antivírus ou o Windows Defender.")
        print("2. Assinatura de Código Legítima:")
        print("   Para distribuição pública oficial, assine o executável com um certificado digital Authenticode")
        print("   usando a ferramenta 'signtool.exe sign /a /tr http://timestamp.digicert.com /td SHA256 EyeMouse.exe'.")
        print("3. Alternativa de Distribuição:")
        print("   Disponibilize o arquivo via GitHub Releases acompanhado de seus hashes criptográficos SHA-256.")
        return True

    except Exception as exc:
        print(f"\nERRO durante o build: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compilador PyInstaller para EyeMouse")
    parser.add_argument(
        "--mode",
        choices=["onedir", "onefile"],
        default="onedir",
        help="Modo de empacotamento: 'onedir' (recomendado para performance) ou 'onefile'",
    )
    cli_args = parser.parse_args()
    success = build(mode=cli_args.mode)
    sys.exit(0 if success else 1)
