# EyeMouse

Controle o mouse do Windows usando apenas seus olhos e uma webcam convencional.

## Requisitos de Sistema

- **Windows** 10 ou superior (64-bit)
- **Python 3.11** (recomendado) ou **3.12**  
  ⚠️ Python 3.13+ **não é suportado** — o MediaPipe não possui wheels oficiais para Python 3.13 no Windows.
- Webcam USB convencional (30+ FPS recomendado)
- CPU razoavelmente recente (a inferência do MediaPipe roda em CPU)

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/lipe404/eyemouse.git
cd eyemouse

# 2. Crie e ative um ambiente virtual com Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\activate

# 3. Instale as dependências
pip install -r eye_mouse/requirements.txt
```

> **Manual de Uso Completo**: Consulte o [Guia do Usuário e Manual de Acessibilidade](docs/USER_GUIDE.md) para detalhes de ergonomia, iluminação, calibração, gestos e atalhos.

## Uso

```bash
python eye_mouse/main.py
```

1. Digite um nome de perfil (ou pressione Enter para "default")
2. Na primeira execução ou via painel, utilize o **Assistente de Configuração Inicial** (Setup Wizard) em 9 etapas:
   - Diagnóstico fotométrico de iluminação facial.
   - Posicionamento da cabeça (50-70 cm da tela).
   - Calibração de olhar com 16 pontos e **validação estrita de holdout** (requer erro < 80px).
   - Escolha do perfil de interação (Híbrido, Dwell ou Gestos).
   - Sessão prática com alvos circulares (Grande 60px, Médio 40px, Pequeno 25px).
3. Use o **Painel de Controle** com abas (`Visão Geral`, `Ajustes Rápidos`, `Avançado`, `Privacidade & Perfis`) para acompanhar FPS, latência e calibração em tempo real.

**Atalho global de emergência:** `Ctrl+Shift+P` — pausa/retoma o controle e aciona liberação imediata de segurança do mouse (*safety release*).

## Modos e Gestos

| Gesto / Ação | Função | Perfil Padrão |
|---|---|---|
| **Olhar para a tela** | Movimenta o cursor (One Euro Filter adaptativo CHI 2012) | Todos |
| **Fixação do olhar (Dwell 900 ms)** | Clique esquerdo automático sem esforço muscular | Híbrido / Dwell |
| **Piscar olho esquerdo** | Clique esquerdo rápido | Híbrido / Gestos |
| **Piscar olho direito** | Clique direito | Híbrido / Gestos |
| **Piscar ambos os olhos** | Duplo clique | Híbrido / Gestos |
| **Fechar olho por > 1.2s** | Iniciar Arrastar e Soltar (*Drag & Drop*) | Híbrido / Gestos |
| **Modo Rolagem (Scroll)** | Olhar para terço superior/inferior da tela rola páginas | Híbrido / Dwell |
| **Barra de Ações (Action Bar)** | Dwell sobre botões para armar clique direito, duplo, arraste ou precisão | Opcional |

## Arquitetura de Baixa Latência

```
Câmera (DirectShow) → CameraCapture (buffer mínimo latest-frame)
                             ↓
                      GazeTracker (MediaPipe Face Landmarker VIDEO mode)
                             ↓
                      HeadPoseEstimator (Compensação 3D de rotação PnP)
                             ↓
                      CalibrationManager (Mapeamento Polinomial + Holdout)
                             ↓
                      OneEuroFilter (Filtragem Adaptativa Sacada vs Fixação)
                             ↓
                      GestureEngine (Arbitragem de piscadas, Dwell & Click Freeze)
                             ↓
                      StateMachine (ACTIVE / PAUSED / TRACKING_LOST / ...)
                             ↓
                      MouseController → OsMouse (Win32 SendInput atômico)
```

- **Sem latência artificial:** o driver usa a API Win32 `SendInput` diretamente, eliminando os 100 ms do `pyautogui.PAUSE` padrão.
- **Click Freeze:** congela coordenadas estáveis durante o fechamento da pálpebra, eliminando desvios indesejados no instante do clique.

## Estrutura do Projeto

```
eye_mouse/
├── main.py                  — Orquestrador principal da aplicação
├── app_state.py             — Máquina de estados AppState thread-safe
├── benchmark.py             — FrameProfiler e medição de latência por estágio
├── os_mouse.py              — Driver Win32 SendInput nativo com DPI awareness
├── camera_capture.py        — Captura desacoplada e buffer de frame mais recente
├── gaze_tracker.py          — Inferência MediaPipe Face Landmarker (modo VIDEO)
├── blink_detector.py        — Detecção de piscada EAR com histerese dupla
├── calibration.py           — Calibração polinomial e validação por holdout
├── mouse_controller.py      — Controlador centralizado de ações do mouse
├── gesture_engine.py        — Motor de gestos, arbitragem e click freeze
├── dwell_clicker.py         — Clique por tempo de fixação com anti-loop
├── scroll_controller.py     — Rolagem vertical por zonas do olhar
├── interaction_profiles.py  — Perfis de interação (Híbrido, Dwell, Gestos)
├── settings_manager.py      — Gerenciador atômico de configurações e privacidade
├── config.py                — Configurações, limiares e feature flags
├── utils/
│   └── smoothing.py         — One Euro Filter, Kalman e Passthrough
└── ui/
    ├── calibration_ui.py    — Interface de calibração em tela cheia
    ├── control_panel.py     — Painel de controle em abas (ttk.Notebook)
    ├── setup_wizard.py      — Assistente de configuração inicial (9 passos)
    ├── training_target_ui.py— Tela de teste e prática com alvos calibrados
    └── action_bar.py        — Barra flutuante de ações com dwell
docs/
├── AUDIT.md                 — Auditoria técnica completa v1.0
├── ROADMAP.md               — Plano de implementação detalhado (M0 a M6)
├── USER_GUIDE.md            — Manual de uso, ergonomia e acessibilidade
└── GESTURE_ENGINE_INTERACTION.md — Especificação técnica do motor de gestos
tests/                       — 345 testes unitários automatizados (pytest)
```

## Executar Testes

```bash
pytest tests/ -v
```

> A suíte completa de **345 testes** roda em ~2 segundos sem necessidade de câmera física ou movimento real do mouse.

## Privacidade e Segurança

- **100% Local na CPU**: Nenhuma gravação de vídeo, foto ou coordenada biométrica é transmitida para a rede.
- **Persistência Mínima**: Apenas parâmetros numéricos de sensibilidade e coeficientes de calibração são armazenados em `~/Documents/EyeMouse/profiles/`.
- **Exclusão de Dados**: Botão disponível no painel para remoção instantânea e permanente de todos os perfis e calibrações locais.

## Limitações Conhecidas

- Precisão de um eye tracker dedicado infravermelho não é garantida com webcam comum.
- Requer Python ≤ 3.12 (limitação de compatibilidade do wheel oficial do MediaPipe no Windows).
- Projetado especificamente para Windows 10/11 (utiliza chamadas SendInput nativas).

## Licença

MIT
