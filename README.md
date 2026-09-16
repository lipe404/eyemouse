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

## Uso

```bash
python eye_mouse/main.py
```

1. Digite um nome de perfil (ou Enter para "default")
2. Siga o processo de calibração de **16 pontos** (grade 4×4)
3. Use o painel de controle flutuante para pausar, recalibrar ou sair

**Atalho global:** `Ctrl+Shift+P` — pausa/retoma o controle

## Gestos

| Gesto | Ação |
|---|---|
| Piscar olho esquerdo | Clique esquerdo |
| Piscar olho direito | Clique direito |
| Piscar os dois | Duplo clique |
| Manter olho fechado (~1,5s) | Iniciar arraste |
| Abrir olho após arraste | Soltar arraste |

## Calibração

O sistema usa **16 pontos** de calibração (grade 4×4). Um subconjunto holdout (≥ 20% dos pontos) é separado antes do treinamento para medir o erro de generalização real. O erro reportado é o **holdout error** — não o resíduo de treinamento.

Se o erro for maior que 80 px, o sistema oferece opção de recalibrar.

## Arquitetura

```
Câmera → frame_queue → GazeTracker (MediaPipe)
                              ↓
                       CalibrationManager (mapeamento polinomial 2ª ordem)
                              ↓
                       SmoothingFilter (Kalman + deadzone)
                              ↓
                       StateMachine (ACTIVE/PAUSED/TRACKING_LOST/...)
                              ↓
                       MouseController → OsMouse (SendInput nativo)
```

**Sem latência artificial:** o driver usa a API Win32 `SendInput` diretamente, eliminando os 100 ms por chamada do `pyautogui.PAUSE` padrão.

## Estrutura do Projeto

```
eye_mouse/
├── main.py               — Orquestrador principal
├── app_state.py          — Máquina de estados (NOVO)
├── benchmark.py          — Sistema de métricas (NOVO)
├── os_mouse.py           — Driver SendInput nativo (NOVO)
├── gaze_tracker.py       — Inferência MediaPipe
├── blink_detector.py     — Detecção de piscada (EAR)
├── calibration.py        — Regressão polinomial + persistência JSON
├── mouse_controller.py   — Controle de alto nível
├── config.py             — Configurações e feature flags
├── utils/smoothing.py    — Filtro de Kalman adaptativo
└── ui/
    ├── calibration_ui.py — UI de calibração (fullscreen)
    └── control_panel.py  — Painel flutuante
docs/
├── AUDIT.md              — Auditoria técnica completa
└── ROADMAP.md            — Plano de implementação em milestones
tests/                    — Testes unitários (pytest)
```

## Executar Testes

```bash
pytest tests/ -v
```

> Os testes não movem o cursor real, não abrem janelas e não requerem câmera.

## Modo Benchmark

Para medir latência de software sem mover o cursor:

```python
# eye_mouse/config.py
BENCHMARK_MODE = True
```

Execute normalmente. O pipeline roda completo mas nenhum evento de mouse é enviado ao SO. O relatório de métricas é salvo no log ao encerrar.

## Persistência

As calibrações são salvas em `~/Documents/EyeMouse/calibration_<perfil>.json`.

- Formato JSON (sem pickle, sem execução de código arbitrário)
- Migração automática de arquivos `.npy` legados
- Versionado (`"version": 2`)

## Feature Flags

Em `config.py`, você pode reverter mudanças individuais:

```python
FT_NATIVE_MOUSE     = True   # OsMouse vs pyautogui
FT_SAFETY_RELEASE   = True   # release_all() em pausas/erros
FT_STATE_MACHINE    = True   # Máquina de estados AppState
FT_JSON_CALIBRATION = True   # Persistência JSON
BENCHMARK_MODE      = False  # Modo benchmark (não move cursor)
```

## Limitações Conhecidas

- Precisão de um eye tracker dedicado não é garantida com webcam convencional
- Requer Python ≤ 3.12 (limitação do MediaPipe)
- `winsound` (beeps de feedback) é exclusivo do Windows

## Licença

MIT
