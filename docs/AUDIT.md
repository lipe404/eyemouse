# EyeMouse — Auditoria Técnica v1.0

**Data:** 2026-09-16  
**Versão auditada:** commit `69d4f0f` (HEAD -> main)  
**Python runtime atual:** 3.13 (MediaPipe exige <= 3.12)

---

## 1. Arquitetura Atual e Fluxo Completo de Dados

### 1.1 Diagrama de Threads

```
Thread principal (Tkinter mainloop)
|
+- camera_loop  [Thread daemon — Produtora]
|    cap.read()
|    frame_queue.put(frame)   <- Queue(maxsize=1), dropa frames velhos
|
+- processing_loop  [Thread daemon — Consumidora]
|    frame_queue.get()
|    cv2.resize(frame, 320x240)
|    GazeTracker.process_frame()
|    |   FaceLandmarker.detect()  <- modo IMAGE (sem tracking temporal)
|    |   Calcula centro da iris (coords normalizadas 0-1)
|    |
|    +-- CalibrationManager.map_to_screen()  <- regressão polinomial 2a ordem
|    +-- SmoothingFilter.update()            <- Kalman 4D + deadzone + lookahead
|    +-- MouseController.move()              <- pyautogui.moveTo()
|    +-- BlinkDetector.process()             <- EAR + historico + derivada
|    +-- MouseController.click/drag()        <- pyautogui.click/mouseDown/Up
|
+- update_ui_loop  [root.after(500ms)]
     ControlPanel.update_status()
```

### 1.2 Pipeline de Dados

| Etapa | Arquivo | Entrada | Saida |
|---|---|---|---|
| Captura | main.py:camera_loop | Webcam 640x480 BGR | frame_queue |
| Resize | main.py:processing_loop | Frame 640x480 | Frame 320x240 |
| Deteccao | gaze_tracker.py | Frame RGB 320x240 | left_iris, right_iris (0-1), landmarks |
| Gaze | main.py | left_iris, right_iris | avg_iris = (left + right) / 2.0 |
| Calibracao | calibration.py | avg_iris (x,y) | (screen_x, screen_y) px |
| Suavizacao | smoothing.py | (screen_x, screen_y) | (smooth_x, smooth_y) |
| Movimento | mouse_controller.py | (smooth_x, smooth_y) | pyautogui.moveTo() |
| Piscada | blink_detector.py | landmarks, img_w, img_h | (l_blink, r_blink, d_blink, hold_start, hold_end, ears) |
| Acao | mouse_controller.py | Evento de piscada | pyautogui.click/mouseDown/mouseUp |

### 1.3 Modulos e Responsabilidades

| Modulo | Papel | Linhas |
|---|---|---|
| config.py | Constantes globais, paths de recursos | 76 |
| gaze_tracker.py | Inferencia MediaPipe, extracao da iris, debug | 144 |
| calibration.py | Coleta, regressao lstsq, save/load, validacao | 180 |
| blink_detector.py | EAR, threshold dinamico, reflexo vs intencional | 284 |
| mouse_controller.py | Wrapper PyAutoGUI + feedback sonoro | 106 |
| utils/smoothing.py | Filtro de Kalman adaptativo, deadzone dinamica | 178 |
| ui/calibration_ui.py | UI fullscreen, animacoes, coleta de gaze | 296 |
| ui/control_panel.py | Painel flutuante, status em tempo real | 149 |
| main.py | Orquestracao, threads, lifecycle | 399 |

---

## 2. Verificacao dos 12 Problemas — Evidencias no Codigo

### P1 — PyAutoGUI com intervalo padrao de 100 ms

**Status: CONFIRMADO**

```python
# mouse_controller.py — linha 23
def __init__(self):
    pyautogui.FAILSAFE = False   # PAUSE nao e zerado aqui
    self.screen_w, self.screen_h = pyautogui.size()
```

`pyautogui.PAUSE` tem padrao 0.1 s (100 ms). Cada chamada de moveTo/click/mouseDown
introduz 100 ms de latencia artificial. A 30 FPS (33 ms/frame), isso equivale a
3 frames perdidos por movimento.

**Impacto: ALTO**

---

### P2 — MediaPipe em modo IMAGE sem rastreamento temporal

**Status: CONFIRMADO**

```python
# gaze_tracker.py — linhas 35-41
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)  # <- Sem running_mode=VIDEO ou LIVE_STREAM
self.detector = vision.FaceLandmarker.create_from_options(options)

# linha 90
detection_result = self.detector.detect(mp_image)
```

Modo IMAGE trata cada frame como imagem independente, sem o filtro temporal interno
do MediaPipe. Modo VIDEO e ate 2x mais estavel para tracking continuo.

**Impacto: ALTO**

---

### P3 — Mapeamento usando posicao absoluta da iris sem compensacao de cabeca

**Status: CONFIRMADO**

```python
# main.py — linhas 346-356
avg_iris = (left_iris + right_iris) / 2.0
# ...
screen_pos = self.calibration_manager.map_to_screen(avg_iris)
```

Coordenadas normalizadas do centro da iris relativas ao frame inteiro. Translacao
fisica da cabeca desloca o cursor mesmo sem mudanca de direcao do olhar.

`output_facial_transformation_matrixes=True` esta ativado mas NUNCA e utilizado
em nenhum trecho do codigo.

**Impacto: ALTO (maior fonte de imprecisao em uso cotidiano)**

---

### P4 — Calibracao pode coletar a mesma amostra repetidamente

**Status: CONFIRMADO**

```python
# calibration_ui.py — linhas 244-256
gaze_data = self.get_latest_gaze()  # le latest_gaze_raw de outra thread
if gaze_data is not None:
    self.calib_manager.add_point(gaze_data, (screen_x, screen_y))
    self.frames_collected += 1
self.window.after(33, lambda: self.collect_loop(point_idx))
```

Se o processamento for mais lento que 33 ms, o mesmo valor de latest_gaze_raw
pode ser lido e adicionado multiplas vezes. Nao ha deduplicacao por timestamp.

**Impacto: MEDIO**

---

### P5 — Validacao da calibracao reutiliza dados de treinamento

**Status: CONFIRMADO**

```python
# calibration.py — linhas 117-125
def _validate_calibration(self):
    for i, iris_pt in enumerate(self.iris_points):   # MESMOS pontos do lstsq
        predicted = self.map_to_screen(iris_pt)
        dist = np.linalg.norm(np.array(predicted) - np.array(screen_pt))
```

Calcula residuo de treinamento, nao erro de generalizacao. Overfitting com
erro de treino baixo mas erro real alto passa despercebido.

**Impacto: MEDIO**

---

### P6 — Botao do mouse nao liberado ao pausar, perder camera ou encerrar

**Status: CONFIRMADO**

```python
# main.py — linha 248
def toggle_pause(self, paused):
    self.is_paused = paused   # NAO chama stop_drag()

# main.py — linha 262
def quit_app(self):
    self.running = False
    self.cap.release()
    sys.exit(0)               # NAO chama stop_drag()

# main.py — linha 295-300 (camera perdida)
self.running = False          # sem stop_drag()
```

Se modo arrastar (mouseDown) estiver ativo, o botao permanece pressionado no SO.
Em tecnologia assistiva, onde o usuario pode nao ter controle dos membros superiores,
isso e um risco critico de usabilidade.

**Impacto: CRITICO**

---

### P7 — Deteccao de piscada depende implicitamente de 30 FPS

**Status: CONFIRMADO**

```python
# blink_detector.py — linha 217
if self.left_closed_frames > (30 * HOLD_DURATION_SEC):
    # constante 30 assume FPS fixo

# config.py — linhas 55-56
BLINK_MIN_FRAMES = 2   # duracao depende do FPS real
BLINK_MAX_FRAMES = 15  # duracao depende do FPS real
```

A 20 FPS: BLINK_MAX_FRAMES=15 equivale a 750 ms.
A 60 FPS: equivale a apenas 250 ms.
Comportamento muda com carga do sistema.

**Impacto: ALTO**

---

### P8 — Suavizacao mistura Kalman, deadzone e predicao de forma nao auditavel

**Status: CONFIRMADO**

```python
# smoothing.py
# Kalman com R adaptativo via sigmoide (linhas 96-117)
# + deadzone dinamica 15px->2px (linhas 131-143)
# + lookahead fixo 66ms (linhas 149-152)
# Nenhuma dessas saidas e logada ou exposta como metrica
```

Nao existe nenhuma metrica de saida que permita medir objetivamente se
uma configuracao produz menos lag ou overshoot.

**Impacto: MEDIO**

---

### P9 — Cursor nao consegue alcancar todas as bordas da tela

**Status: CONFIRMADO**

```python
# mouse_controller.py — linhas 60-61
final_x = max(SCREEN_MARGIN, min(self.screen_w - SCREEN_MARGIN, smooth_x))
final_y = max(SCREEN_MARGIN, min(self.screen_h - SCREEN_MARGIN, smooth_y))

# config.py — linha 71
SCREEN_MARGIN = 50  # clamp bilateral de 50px em todos os lados
```

50 px em todos os lados = ~15% da area da tela inacessivel, incluindo
barra de tarefas, botoes de fechar janela e menus de borda.

**Impacto: ALTO**

---

### P10 — Persistencia com allow_pickle=True

**Status: CONFIRMADO**

```python
# calibration.py — linha 170
data = np.load(self.calibration_file, allow_pickle=True).item()
```

allow_pickle=True desativa protecao contra execucao de codigo arbitrario
durante desserializacao. Salvar dicionario com np.save usa pickle
implicitamente, criando fragilidade entre versoes do NumPy.

**Impacto: BAIXO (seguranca teorica em uso pessoal), MEDIO (compatibilidade)**

---

### P11 — Estado congelado sem notificacao ao perder o rosto

**Status: CONFIRMADO**

```python
# main.py — linhas 342-349
if left_iris is not None and right_iris is not None:
    avg_iris = (left_iris + right_iris) / 2.0
    with self.data_lock:
        self.latest_gaze_raw = avg_iris   # so atualiza SE rosto detectado
```

Quando rosto desaparece, latest_gaze_raw mantem o ultimo valor. A CalibrationUI
continua coletando esse valor congelado como dado valido. Nenhum indicador
visual notifica o usuario da perda de rastreamento.

**Impacto: MEDIO**

---

### P12 — Processing_loop monolitico e excessivamente acoplado

**Status: CONFIRMADO**

```python
# main.py — linhas 315-394 (~80 linhas)
def processing_loop(self):
    # leitura de fila + resize + inferencia + gaze + calibracao +
    # suavizacao + movimento de cursor + piscada + acao de mouse +
    # FPS + UI — TUDO no mesmo metodo
```

Impossivel: testar pipeline sem acionar mouse real; adicionar gestos sem
modificar o metodo central; medir tempo de cada etapa; desacoplar controle
com base em confianca.

**Impacto: MEDIO (manutencao e evolucao)**

---

## 3. Problemas Adicionais Encontrados na Auditoria

| # | Problema | Arquivo | Impacto |
|---|---|---|---|
| P13 | Arquivo espu'rio `et --soft HEAD~4'` na raiz (saida de git log redirecionada) | raiz | Baixo |
| P14 | face_landmarker.task duplicado (raiz e eye_mouse/) | raiz | Baixo |
| P15 | README cita "9 pontos" mas config.py usa 16 | README.md, config.py:65 | Baixo |
| P16 | Ausencia de espelhamento horizontal da camera | main.py | Medio |
| P17 | blink_detector recebe img_w/h de 640x480 mas landmarks sao de 320x240 — nao e bug (EAR e adimensional) mas e confuso, merece comentario | main.py:364 | Baixo |
| P18 | Python <= 3.12 nao documentado em lugar nenhum | README.md, requirements.txt | Alto |
| P19 | sys.path.insert repetido em cada arquivo de teste (conftest.py ja faz isso) | tests/*.py | Baixo |

---

## 4. Causas Provaveis de Latencia, Imprecisao e Falsos Cliques

### 4.1 Latencia de Cursor

| Causa | Magnitude | Arquivo |
|---|---|---|
| pyautogui.PAUSE = 0.1s (padrao) | +100 ms por chamada | mouse_controller.py |
| Inferencia MediaPipe (modo IMAGE) | 8-25 ms/frame | gaze_tracker.py |
| Lookahead de predicao fixo | +66 ms injetado no calculo | smoothing.py |
| Deadzone passiva (15 px repouso) | Micro-movimentos ignorados | smoothing.py |

Maior impacto: pyautogui.PAUSE. Setar pyautogui.PAUSE = 0 pode reduzir
latencia percebida em 60-70%.

### 4.2 Imprecisao de Mapeamento

| Causa | Efeito |
|---|---|
| Media das iris sem compensacao de pose | Cursor deriva com translacao da cabeca |
| Validacao usa dados de treino | Aceita calibracoes ruins como boas |
| Amostras duplicadas na calibracao | Enviesamento do modelo |
| Regressao 2a ordem sem holdout | Pode extrapolar mal nas bordas |

### 4.3 Falsos Cliques

| Causa | Efeito |
|---|---|
| BLINK_REFLEX_OPEN_SPEED tolerante em fadiga ocular | Reflexos passam como cliques |
| FPS instavel amplia BLINK_MAX_FRAMES | Piscadas lentas capturadas em CPUs lentas |
| latest_gaze_raw congelado durante perda de rosto | Calibracao contaminada por amostras fantasmas |

---

## 5. Dependencias e Compatibilidade Windows

| Biblioteca | Compatibilidade Windows | Notas |
|---|---|---|
| Python 3.11 | Recomendado | MediaPipe suportado, estavel |
| Python 3.12 | OK | MediaPipe suportado |
| Python 3.13 | NAO suportado pelo MediaPipe | Sem wheel oficial no pip |
| MediaPipe 0.10.x | OK (3.9-3.12) | CPU only no Windows |
| OpenCV 4.x | OK | Captura com DirectShow |
| PyAutoGUI 0.9.54 | OK | Requer pywin32 |
| keyboard 0.13.5 | OK | Pode exigir admin para hotkeys globais |
| winsound | OK (Windows only) | try/except ja presente no codigo |
| tkinter | OK (built-in) | |
| PyInstaller 6.x | OK | --onefile lento na primeira execucao |
| NumPy 1.x/2.x | OK | allow_pickle fragil entre versoes |

---

## 6. Metricas de Estado Inicial (Baseline)

| Metrica | Como Medir | Valor Estimado |
|---|---|---|
| Latencia cursor (input->screen) | time.time() antes/apos moveTo() | ~130-200 ms |
| FPS de processamento real | Contador em processing_loop | ~20-30 FPS |
| Erro de reprojecao pos-calibracao | Log de compute_calibration() | 30-80 px |
| Taxa de falsos cliques (5 min) | Contagem manual durante uso | A medir |
| Drift com translacao de cabeca 1 cm | Deslocamento do cursor em pixels | A medir |

---

## 7. Arquitetura Proposta para Versao Otimizada

### 7.1 Principios

1. Separacao de concerns estrita: pipeline desacoplado de acoes de mouse.
2. Seguranca first: liberar botao do mouse em qualquer transicao de estado.
3. Metricas em toda saida critica.
4. Incremental e reversivel: feature toggles e testes antes/depois.

### 7.2 Estrutura de Modulos Proposta

```
eye_mouse/
+-- config.py                     (sem mudancas estruturais)
+-- gaze_tracker.py               (-> modo VIDEO, compensacao de pose)
+-- calibration.py                (-> holdout validation, JSON storage)
+-- blink_detector.py             (-> timing em tempo real, nao frames)
+-- mouse_controller.py           (-> PAUSE=0, liberacao centralizada)
+-- gesture_engine.py             [NOVO] interpretacao de gestos
+-- pipeline.py                   [NOVO] orquestracao sem acoplamento
+-- utils/
|   +-- smoothing.py              (-> metricas de saida)
|   +-- pose_compensation.py      [NOVO] subtrai translacao/rotacao da cabeca
|   +-- metrics.py                [NOVO] coleta de metricas runtime
+-- ui/
    +-- calibration_ui.py         (-> deduplicacao de amostras por timestamp)
    +-- control_panel.py          (-> indicador de perda de face)
```

### 7.3 Fluxo de Dados Proposto

```
camera_loop -> frame_queue
                  |
           GazeTracker (modo VIDEO)
                  |
           PoseCompensator  <-- subtrai translacao/rotacao da cabeca
                  |
           CalibrationManager.map_to_screen()
                  |
           SmoothingFilter (+ metricas)
                  |
           GestureEngine   <-- eventos discretos (blink, hold, scroll)
                  |
           EventBus        <-- desacopla deteccao de acao
                  |
           MouseController (pyautogui.PAUSE = 0, liberacao segura)
```

---

## 8. Resumo de Problemas por Prioridade

### CRITICO

| # | Problema | Arquivo | Linha |
|---|---|---|---|
| P1 | pyautogui.PAUSE=0.1s nao desativado | mouse_controller.py | 23 |
| P6 | Botao mouse nao liberado ao pausar/encerrar | main.py | 248, 262 |
| P3 | Gaze absoluto sem compensacao de pose de cabeca | main.py | 346 |

### ALTO

| # | Problema | Arquivo | Linha |
|---|---|---|---|
| P7 | Timing de piscada em frames, nao segundos reais | blink_detector.py | 217 |
| P2 | MediaPipe em modo IMAGE sem tracking temporal | gaze_tracker.py | 35 |
| P9 | Cursor nao alcanca bordas (clamp 50px bilateral) | mouse_controller.py | 60 |
| P11 | Estado congelado sem notificacao ao perder rosto | main.py | 342 |
| P18 | Python <= 3.12 nao documentado | README.md | - |

### MEDIO

| # | Problema | Arquivo | Linha |
|---|---|---|---|
| P4 | Amostras duplicadas na calibracao | calibration_ui.py | 244 |
| P5 | Validacao usa dados de treino | calibration.py | 101 |
| P8 | Suavizacao nao auditavel | smoothing.py | - |
| P10 | allow_pickle=True no load | calibration.py | 170 |
| P12 | processing_loop monolitico | main.py | 315 |
| P16 | Sem espelhamento horizontal da camera | main.py | - |

### BAIXO

| # | Problema | Arquivo |
|---|---|---|
| P13 | Arquivo espu'rio na raiz | raiz |
| P14 | face_landmarker.task duplicado | raiz |
| P15 | README desatualizado (9 vs 16 pontos) | README.md |
| P17 | Comentario faltando sobre img_w/h vs resolucao de deteccao | main.py:364 |
| P19 | sys.path.insert redundante nos testes | tests/*.py |
