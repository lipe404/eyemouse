# EyeMouse — Roadmap de Implementacao

**Data:** 2026-09-16
**Versao de referencia:** Auditoria v1.0 (docs/AUDIT.md)
**Objetivo:** Transformar o prototipo em aplicacao utilizavel no cotidiano,
de forma incremental, preservando testes existentes e possibilitando reversao.

---

## Principios Gerais

- Cada milestone tem ENTRADAS (pre-condicoes), SAIDAS (o que entrega)
  e CRITERIOS DE ACEITACAO mensuráveis.
- Nenhuma mudanca sem cobertura de teste (novo ou existente).
- Feature flags via config.py permitem ativar/desativar mudancas
  sem alterar logica de producao.
- Pull requests por milestone; sem mega-commits.
- Medicao de baseline ANTES de cada milestone e verificacao DEPOIS.

---

## Milestone 0 — Baseline e Limpeza (Esta Etapa)

**Objetivo:** Estado inicial documentado, repositorio limpo, metricas definidas.
**Status:** CONCLUÍDO (2026-09-16)

### Tarefas

- [x] Auditoria tecnica completa (docs/AUDIT.md)
- [x] Roadmap de implementacao (docs/ROADMAP.md)
- [x] Remover arquivo espu'rio: git rm "et --soft HEAD~4'"
- [x] Remover face_landmarker.task duplicado da raiz
- [x] Atualizar README.md: 16 pontos, Python 3.11/3.12, instrucoes venv
- [x] Instrumentar processing_loop com medicao de FPS real e latencia (benchmark.py)
- [x] Script de benchmark interno e FrameProfiler implementado

### Criterios de Aceitacao

- [x] docs/AUDIT.md e docs/ROADMAP.md presentes e revisados.
- [x] Repositorio sem arquivos espurios.
- [x] README atualizado e correto.
- [x] FrameProfiler e modo benchmark integrados.

---

## Milestone 1 — Correcoes Criticas de Seguranca e Latencia

**Objetivo:** Corrigir os tres problemas criticos (P1, P6, P3-parcial).
**Dependencias:** Milestone 0 concluido, baseline medido.
**Estimativa:** 1-2 dias de desenvolvimento.
**Status:** CONCLUÍDO (2026-09-16)

### 1.1 — Desativar pyautogui.PAUSE (P1)

**Arquivo:** eye_mouse/mouse_controller.py

**Mudanca:**

```python
# ANTES (linha 23)
def __init__(self):
    pyautogui.FAILSAFE = False

# DEPOIS
def __init__(self):
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0   # Remove 100ms de latencia por chamada
```

**Metrica:** Medir latencia media de moveTo() antes e depois.
Esperado: reducao de ~100 ms para ~1-3 ms por chamada.

**Risco:** Nenhum. PAUSE e apenas um sleep artificial.
**Reversao:** Remover a linha adicionada.
**Teste:** test_mouse_controller.py — adicionar assertion de pyautogui.PAUSE == 0.0.

---

### 1.2 — Liberacao Centralizada do Botao do Mouse (P6)

**Arquivo:** eye_mouse/mouse_controller.py e eye_mouse/main.py

**Mudanca em mouse_controller.py:** adicionar metodo safety_release():

```python
def safety_release(self):
    """Libera qualquer botao pressionado. Chamar em pausas, erros e encerramento."""
    if self.is_dragging:
        pyautogui.mouseUp()
        self.is_dragging = False
        self._play_sound(300, 100)  # Som grave de liberacao de emergencia
```

**Mudanca em main.py:** chamar safety_release() em:

- toggle_pause() quando paused=True
- quit_app() antes de sys.exit()
- camera_loop() antes de self.running = False (camera perdida)

**Metrica:** Teste manual — ativar modo arrastar, pressionar Ctrl+Shift+P;
verificar que o botao e liberado (titulo da janela nao mostra "movendo").

**Risco:** Baixo. Apenas adiciona chamada defensiva em eventos ja existentes.
**Reversao:** Remover as chamadas a safety_release().
**Teste:** Adicionar test_safety_release() em test_mouse_controller.py e
test_main.py verificando que stop_drag e chamado nos eventos relevantes.

---

### 1.3 — Indicador de Perda de Face (P11 — parte de P3)

**Arquivo:** eye_mouse/main.py e eye_mouse/ui/control_panel.py

**Mudanca em main.py:** expor estado de deteccao de face:

```python
# processing_loop
face_detected = left_iris is not None and right_iris is not None
with self.data_lock:
    self.face_detected = face_detected
    if face_detected:
        self.latest_gaze_raw = avg_iris
        self.last_face_time = current_time
```

**Mudanca em control_panel.py:** adicionar label de status de face:

```python
self.face_status_label = ttk.Label(status_frame, text="Rosto: Detectado",
                                    foreground="green")
```

**Mudanca em main.py:update_ui_loop:** atualizar label de face.

**Metrica:** Ao cobrir a camera, o label deve mudar para "Rosto: Perdido"
em menos de 500 ms.

**Risco:** Baixo. Apenas adiciona informacao de status na UI.
**Reversao:** Remover label e atualizacao correspondente.

### Criterios de Aceitacao do Milestone 1

- pyautogui.PAUSE == 0.0 em producao.
- Botao do mouse e liberado em 100% dos cenarios de pausa/encerramento/camera perdida.
- Label de status de rosto reflete deteccao em tempo real.
- Todos os testes existentes passam sem modificacao.
- Latencia de moveTo() medida: < 10 ms por chamada (vs ~100 ms antes).

---

## Milestone 2 — Estabilidade da Deteccao de Piscada e Camera

**Objetivo:** Corrigir P7 (timing FPS-dependente) e P2 (modo MediaPipe).
**Dependencias:** Milestone 1 concluido.
**Estimativa:** 2-3 dias.

### 2.1 — Timing de Piscada Baseado em Tempo Real (P7)

**Arquivo:** eye_mouse/blink_detector.py e eye_mouse/config.py

**Mudanca em config.py:** adicionar parametros de tempo real:

```python
# Remover (ou manter como fallback comentado):
# BLINK_MIN_FRAMES = 2
# BLINK_MAX_FRAMES = 15

# Adicionar:
BLINK_MIN_DURATION_SEC = 0.066   # ~2 frames a 30 FPS
BLINK_MAX_DURATION_SEC = 0.500   # ~15 frames a 30 FPS
HOLD_DURATION_SEC = 1.5          # ja existente — usar com timestamp
```

**Mudanca em blink_detector.py:** substituir contadores de frames por timestamps:

```python
# ANTES
self.left_closed_frames = 0
# ...
if self.left_closed_frames > (30 * HOLD_DURATION_SEC):

# DEPOIS
self.left_closed_start_time = None
# ...
if left_closed:
    if self.left_closed_start_time is None:
        self.left_closed_start_time = current_time
    duration = current_time - self.left_closed_start_time
    if duration >= HOLD_DURATION_SEC and not self.is_holding:
        self.is_holding = True
        hold_start = True
else:
    if self.left_closed_start_time is not None:
        duration = current_time - self.left_closed_start_time
        if BLINK_MIN_DURATION_SEC <= duration <= BLINK_MAX_DURATION_SEC:
            # ... logica de clique
        self.left_closed_start_time = None
```

**Metrica:** Com FPS variando entre 15 e 60, a duracao minima de piscada
deve permanecer 66 ms ± 10 ms (medida com timestamps reais).

**Risco:** Medio. Muda logica central da deteccao de piscadas.
**Mitigacao:** Feature flag BLINK_USE_REALTIME_TIMING = True em config.py.
Testes existentes em test_blink_detector.py devem ser adaptados, nao removidos.

---

### 2.2 — MediaPipe em Modo VIDEO (P2)

**Arquivo:** eye_mouse/gaze_tracker.py

**Mudanca:**

```python
# ANTES
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
self.detector = vision.FaceLandmarker.create_from_options(options)
# Chamada: self.detector.detect(mp_image)

# DEPOIS
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
self.detector = vision.FaceLandmarker.create_from_options(options)
self._frame_timestamp_ms = 0
# Chamada: self.detector.detect_for_video(mp_image, timestamp_ms)
```

Em modo VIDEO, o MediaPipe aplica filtro temporal interno entre frames,
reduzindo jitter dos landmarks da iris.

**Nota:** O modo VIDEO requer que os timestamps sejam crescentes e
em milissegundos. Usar int(time.time() * 1000).

**Metrica:** Medir variancia das coordenadas da iris com rosto estatico
antes e depois. Esperado: reducao de pelo menos 30% na variancia.

**Risco:** Medio. Muda a API de chamada do detector.
**Mitigacao:** Feature flag GAZE_USE_VIDEO_MODE = True.
Teste de integracao com frame sintetico.

### Criterios de Aceitacao do Milestone 2

- Piscada detectada consistentemente entre 66ms e 500ms independente do FPS.
- Hold de 1.5 s detectado com erro < 100 ms em qualquer FPS.
- Variancia das coordenadas da iris com cabeca estatica reduzida >= 30%.
- Todos os testes existentes adaptados e passando.

---

## Milestone 3 — Qualidade da Calibracao

**Objetivo:** Corrigir P4 (amostras duplicadas), P5 (validacao em treino),
P9 (clamp de bordas) e P10 (allow_pickle).
**Dependencias:** Milestone 1 concluido.
**Estimativa:** 2-3 dias.

### 3.1 — Deduplicacao de Amostras (P4)

**Arquivo:** eye_mouse/ui/calibration_ui.py e eye_mouse/main.py

**Estrategia:** adicionar timestamp de ultima atualizacao de gaze_raw:

```python
# main.py — processing_loop
with self.data_lock:
    self.latest_gaze_raw = avg_iris
    self.latest_gaze_timestamp = current_time   # NOVO

# calibration_ui.py — collect_loop
gaze_data = self.get_latest_gaze()
gaze_ts = self.get_latest_gaze_timestamp()   # NOVO

if gaze_data is not None and gaze_ts != self._last_collected_ts:
    self._last_collected_ts = gaze_ts         # NOVO
    self.calib_manager.add_point(gaze_data, (screen_x, screen_y))
    self.frames_collected += 1
```

**Metrica:** Durante calibracao, verificar que nenhum par (iris, screen)
identico aparece duas vezes consecutivas no iris_points.

---

### 3.2 — Validacao com Holdout (P5)

**Arquivo:** eye_mouse/calibration.py

**Estrategia:** separar 20% dos pontos para validacao antes do treinamento:

```python
def compute_calibration(self):
    n = len(self.iris_points)
    if n < 6:
        return False, 0.0
  
    # Separar holdout (20% ou min 2 pontos)
    n_holdout = max(2, n // 5)
    indices = list(range(n))
    holdout_idx = indices[-n_holdout:]   # ultimos N pontos como holdout
    train_idx = indices[:-n_holdout]
  
    # Treinar com train_idx
    # Validar com holdout_idx
    # Retornar erro de holdout (nao de treino)
```

**Metrica:** O erro reportado deve ser o holdout error.
Com calibracao de 16 pontos: 3 pontos de holdout, 13 de treino.

---

### 3.3 — Remover Clamp Bilateral e Ajustar Margens (P9)

**Arquivo:** eye_mouse/config.py e eye_mouse/mouse_controller.py

**Estrategia:**

- Reduzir SCREEN_MARGIN de 50 para 0 (ou deixar configuravel).
- Manter apenas clamp nos limites absolutos da tela (0 e screen_w/h).
- Ajustar pontos de calibracao para incluir pontos nas bordas extremas.

```python
# config.py
SCREEN_MARGIN = 0   # Ou parametro configuravel pelo usuario

# mouse_controller.py
final_x = max(0, min(self.screen_w - 1, smooth_x))
final_y = max(0, min(self.screen_h - 1, smooth_y))
```

---

### 3.4 — Migrar Persistencia de .npy para JSON (P10)

**Arquivo:** eye_mouse/calibration.py

**Estrategia:** salvar coeficientes como JSON (array de floats):

```python
import json

def save_calibration(self):
    if self.is_calibrated:
        data = {
            "coeffs_x": self.coeffs_x.tolist(),
            "coeffs_y": self.coeffs_y.tolist(),
            "version": 1,
        }
        path = self.calibration_file.replace(".npy", ".json")
        with open(path, "w") as f:
            json.dump(data, f)

def load_calibration(self):
    path = self.calibration_file.replace(".npy", ".json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        self.coeffs_x = np.array(data["coeffs_x"])
        self.coeffs_y = np.array(data["coeffs_y"])
        self.is_calibrated = True
        return True
    # Fallback para .npy legado
    if os.path.exists(self.calibration_file):
        # carregar e migrar automaticamente
        ...
```

**Compatibilidade:** manter fallback de leitura do .npy para usuarios
com calibracoes existentes; migrar automaticamente ao salvar.

### Criterios de Aceitacao do Milestone 3

- Nenhuma amostra duplicada em calibracoes com processamento lento (simulado).
- Erro reportado e o holdout error (verificar em log).
- Cursor alcanca as 4 bordas da tela durante teste manual.
- Arquivo de calibracao salvo como .json, legivel e sem pickle.
- Leitura de arquivo .npy legado funciona para usuarios existentes.

---

## Milestone 4 — Compensacao de Pose de Cabeca

**Objetivo:** Reduzir drift do cursor causado por translacao da cabeca (P3 completo).
**Dependencias:** Milestone 2 concluido (modo VIDEO disponivel).
**Estimativa:** 3-5 dias. Maior complexidade tecnica do roadmap.

### Estrategia

**Opção A (Simples — Baseada em Landmarks):**
Normalizar a posicao da iris relativa aos cantos anatomicos do proprio olho
(lacrimal medial e canto lateral), em vez de relativa ao frame inteiro.

```python
# utils/pose_compensation.py
def normalize_iris_to_eye(landmarks, iris_center, eye_corner_indices):
    """
    Retorna posicao relativa da iris dentro do olho.
    iris_center: coords normalizadas [0,1] no frame
    Retorna: (u, v) onde 0=canto esq, 1=canto dir do olho
    """
    inner = np.array([landmarks[eye_corner_indices[0]].x,
                      landmarks[eye_corner_indices[0]].y])
    outer = np.array([landmarks[eye_corner_indices[1]].x,
                      landmarks[eye_corner_indices[1]].y])
    eye_width = np.linalg.norm(outer - inner)
    if eye_width < 1e-6:
        return iris_center
    rel = (iris_center - inner) / eye_width
    return rel
```

**Opcao B (Robusta — Via Matriz de Pose):**
Usar as `facial_transformation_matrixes` do MediaPipe para obter
a rotacao e translacao da cabeca e subtrair do vetor de olhar.

**Recomendacao:** Implementar Opcao A primeiro (mais simples, menor risco).
Medir impacto. Se insuficiente, escalar para Opcao B.

**Metrica:**

- Mover cabeca 2 cm lateralmente com olhar fixo em ponto central.
- Medir deslocamento do cursor antes e depois.
- Meta: reducao de > 70% no drift.

**Risco:** Alto. Muda a representacao fundamental do gaze.
**Mitigacao:** Feature flag GAZE_POSE_COMPENSATION = True.
Re-calibrar apos ativar (obrigatorio — o espaco de features muda).

### Criterios de Aceitacao do Milestone 4

- Drift com translacao lateral de 3 cm: < 50 px na tela.
- Qualidade de calibracao mantida ou melhorada (holdout error).
- Todos os testes adaptados e passando.

---

## Milestone 5 — Desacoplamento do Pipeline e Gestos

**Objetivo:** Resolver P12 (acoplamento) e preparar para novos gestos (scroll, etc.).
**Dependencias:** Milestones 1-4 concluidos.
**Estimativa:** 3-4 dias.

### 5.1 — Extrair GestureEngine

**Novo arquivo:** eye_mouse/gesture_engine.py

Responsavel apenas por: receber eventos de blink_detector e emitir
comandos de alto nivel (CLICK_LEFT, CLICK_RIGHT, DOUBLE_CLICK,
DRAG_START, DRAG_END, SCROLL_UP, SCROLL_DOWN).

### 5.2 — Extrair Pipeline

**Novo arquivo:** eye_mouse/pipeline.py

```python
class GazePipeline:
    def process(self, frame) -> PipelineResult:
        """Retorna dados de gaze, confianca e eventos de gesto."""
        ...

@dataclass
class PipelineResult:
    screen_pos: Optional[Tuple[int, int]]
    confidence: float         # 0.0-1.0
    gesture_events: List[str] # ["CLICK_LEFT", "DRAG_START", ...]
    face_detected: bool
    fps: float
```

### 5.3 — Simplificar main.py

processing_loop vira apenas:

```python
def processing_loop(self):
    while self.running:
        frame = self.frame_queue.get(timeout=1.0)
        result = self.pipeline.process(frame)
        self._apply_result(result)

def _apply_result(self, result: PipelineResult):
    if not self.is_paused and result.screen_pos:
        self.mouse_controller.move(*result.screen_pos)
    for event in result.gesture_events:
        self._dispatch_gesture(event)
```

### Criterios de Aceitacao do Milestone 5

- Todos os comportamentos existentes preservados.
- processing_loop com < 20 linhas de logica de negocio.
- GestureEngine testavel sem instanciar EyeMouseApp.
- Cobertura de testes >= cobertura anterior (sem regressao).

---

## Milestone 6 — Novos Gestos e Funcionalidades

**Objetivo:** Adicionar funcionalidades solicitadas no objetivo do projeto.
**Dependencias:** Milestone 5 concluido (GestureEngine disponivel).
**Estimativa:** 4-6 dias.

### 6.1 — Scroll por Gaze (rolar paginas)

**Estrategia:**

- Zona de scroll superior/inferior da tela (ex: 10% borda superior/inferior).
- Quando cursor entra nessas zonas por > 500 ms, aciona scroll.
- Ou: piscar os dois olhos rapidamente dispara scroll (novo gesto no GestureEngine).

### 6.2 — Perfis de Interacao

**Estrategia:**

- Salvar conjuntos de parametros (EMA_ALPHA, BLINK_EAR_THRESHOLD, etc.) por nome.
- UI no ControlPanel para selecionar perfil ativo.
- Arquivo de perfis: ~/Documents/EyeMouse/profiles.json

### 6.3 — Espelhamento Horizontal Configuravel (P16)

```python
# config.py
MIRROR_CAMERA = False  # Inverter horizontalmente o frame da camera

# main.py:camera_loop
if MIRROR_CAMERA:
    frame = cv2.flip(frame, 1)
```

### 6.4 — Indicador de Confianca Visual

Quando confianca do gaze < 0.5 (ex: landmarks instáveis), exibir cursor
semitransparente ou cor diferente via ControlPanel.

### Criterios de Aceitacao do Milestone 6

- Scroll funciona em pelo menos 3 aplicativos diferentes (browser, VSCode, explorer).
- Troca de perfil sem reiniciar a aplicacao.
- Todos os novos gestos testados e documentados.

---

## Tabela de Metricas por Milestone

| Milestone | Metrica Principal            | Meta                   | Como Medir                     |
| --------- | ---------------------------- | ---------------------- | ------------------------------ |
| M0        | Baseline documentado         | ---                    | Script de benchmark            |
| M1        | Latencia de moveTo()         | < 10 ms                | time.time() antes/apos         |
| M1        | Seguranca de liberacao       | 100% dos casos         | Teste manual + automatizado    |
| M2        | Timing de piscada            | +/-10 ms de 66-500 ms  | Timestamps reais               |
| M2        | Variancia iris estatica      | Reducao >= 30%         | Coleta de 100 frames estaticos |
| M3        | Holdout error calibracao     | Reportado com precisao | Log estruturado                |
| M3        | Cursor em bordas             | 100% area acessivel    | Teste manual em 4 bordas       |
| M4        | Drift por translacao 3cm     | < 50 px                | Medicao com regua fisica       |
| M5        | Complexidade processing_loop | < 20 linhas negocio    | Contagem de linhas             |
| M6        | Scroll funcional             | 3 apps diferentes      | Teste manual documentado       |

---

## Estrategia de Medicao e Reversao

### Medicao Objetiva

Para cada milestone, executar o script de benchmark antes e depois:

```
python eye_mouse/utils/benchmark.py --duration 60 --output results/M{N}_{before|after}.json
```

O script coleta: FPS medio, latencia min/max/media de moveTo(),
taxa de frames processados vs descartados, uso de CPU/RAM.

### Reversao

Cada mudanca usa feature flags em config.py:

```python
# FEATURES (definir como False para reverter)
FT_PYAUTOGUI_PAUSE_ZERO = True      # M1.1
FT_SAFETY_RELEASE = True            # M1.2
FT_REALTIME_BLINK_TIMING = True     # M2.1
FT_VIDEO_MODE_MEDIAPIPE = True      # M2.2
FT_CALIBRATION_DEDUP = True         # M3.1
FT_HOLDOUT_VALIDATION = True        # M3.2
FT_SCREEN_MARGIN_ZERO = True        # M3.3
FT_JSON_CALIBRATION = True          # M3.4
FT_POSE_COMPENSATION = True         # M4
FT_GESTURE_ENGINE = True            # M5
FT_MIRROR_CAMERA = False            # M6.3 (desabilitado por padrao)
```

### Riscos de Regressao e Medidas

| Risco                                                                      | Probabilidade | Impacto | Mitigacao                                                              |
| -------------------------------------------------------------------------- | ------------- | ------- | ---------------------------------------------------------------------- |
| Modo VIDEO do MediaPipe incompativel com versoes antigas                   | Medio         | Alto    | Testar em MediaPipe 0.10.x; fallback para modo IMAGE                   |
| Compensacao de pose piora calibracao para usuarios sem movimento de cabeca | Medio         | Medio   | Feature flag; comparar holdout error antes/depois                      |
| pyautogui.PAUSE=0 causa race condition em acoes rapidas                    | Baixo         | Medio   | Testar cliques rapidos multiplos; adicionar lock interno se necessario |
| Mudanca de formato .npy para .json quebra calibracoes existentes           | Alto          | Medio   | Fallback de leitura .npy implementado e testado                        |
| Remocao de SCREEN_MARGIN causa cursor preso nas bordas da tela             | Baixo         | Baixo   | Clamp em 0 e screen_w-1 permanece                                      |

---

## Proximos Passos Imediatos (Apos Aprovacao deste Documento)

1. Executar Milestone 0 (limpeza) — < 1 dia.
2. Medir baseline com script de benchmark.
3. Implementar M1.1 (pyautogui.PAUSE=0) — mudanca de 1 linha, maximo impacto imediato.
4. Implementar M1.2 (safety_release) — seguranca critica para tecnologia assistiva.
5. Implementar M1.3 (indicador de face) — feedback visual sem risco.
6. Revisar, testar e medir antes de avancar para M2.
