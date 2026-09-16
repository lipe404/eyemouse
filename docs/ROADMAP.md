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

**Objetivo:** Otimizar captura de vídeo, processamento de imagem, modos do MediaPipe, validação temporal e mitigação de latência.
**Dependencias:** Milestone 1 concluido.
**Estimativa:** 2-3 dias.
**Status:** CONCLUÍDO (2026-09-16)

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
**Status:** CONCLUÍDO (2026-09-16)

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

## Milestone 4 — Cursor Fluido, Responsivo e Preciso

**Objetivo:** Transformar o controle do cursor em um sistema fluido, responsivo e estável, eliminando o overshoot e o jitter em repouso sem gerar lag perceptível em sacadas.
**Dependencias:** Milestones 1, 2 e 3 concluídos.
**Status:** CONCLUÍDO (2026-09-16)

### 4.1 — Filtros Intercambiáveis e One Euro Filter
- Implementação fiel do **One Euro Filter** (Casiez et al., CHI 2012) adaptativo à velocidade instantânea (`min_cutoff=1.0`, `beta=0.007`, `d_cutoff=1.0`).
- Preservação do `KalmanSmoothingFilter` (OpenCV) como benchmark de referência.
- Remoção do lookahead fixo de 66 ms por padrão (eliminando overshoot de 42,2 px / 4,3% em sacadas).
- Introdução do `PassThroughFilter` (sem filtro) como baseline de latência zero.
- Factory `create_smoothing_filter(filter_type)` e alternância em runtime.

### 4.2 — Controlador de Cursor de Alta Resolução
- Separação estrita dos estágios de coordenadas:
  1. `last_raw_pos`: float do modelo de calibração.
  2. `last_filtered_pos`: float do filtro de suavização.
  3. `last_sent_pos`: int enviado ao driver do SO.
- Sem arredondamento prematuro: precisão float preservada ao longo de todo o pipeline.
- Alcance completo dos 4 cantos da tela com `SCREEN_MARGIN = 0`.
- **Modo de Precisão (Precision Mode):** atenuação de ganho em torno de uma âncora (`precision_factor = 0.35`) para mira milimétrica em botões pequenos e links de texto.

### 4.3 — Backend Windows Nativo com DPI Awareness e Multi-Monitor
- Declaração de **DPI Awareness** no Windows (`SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)`) com fallbacks seguros, garantindo pixels físicos 1:1.
- Suporte a **Desktop Virtual Multi-Monitor** com coordenadas negativas (`SM_XVIRTUALSCREEN`, `SM_YVIRTUALSCREEN`, `MOUSEEVENTF_VIRTUALDESK`).
- Abstração `BaseMouseBackend` e suporte a `PyAutoGuiMouse` como driver alternativo de teste.

### 4.4 — Benchmark e Documentação
- Script automatizado `eye_mouse/benchmark_smoothing.py` com cenários sintéticos de fixação, sacada e perseguição suave.
- Documentação técnica abrangente em `docs/CURSOR_SMOOTHING_BACKENDS.md`.

### Critérios de Aceitação do Milestone 4
- [x] Redução de jitter em fixação > 50% sem criar lag mecânico perceptível (60,4% obtido).
- [x] Eliminação do overshoot do lookahead antigo (overshoot caiu de 42,2 px para 1,9 px).
- [x] Suporte a multi-monitor e DPI awareness ativo.
- [x] 100% de retrocompatibilidade: 264 testes passando com zero falhas.

---

## Milestone 5 — Motor de Gestos, Interação Completa e Acessibilidade Universal

**Objetivo:** Permitir o controle hands-free completo do Windows (clique esquerdo, direito, duplo, arraste, rolagem, precisão, pausa) por gestos voluntários e permanência (Dwell), com acessibilidade universal.
**Dependencias:** Milestones 1-4 concluídos.
**Status:** CONCLUÍDO (2026-09-16)

### 5.1 — Máquina de Estados de Piscadas e Histerese (`BlinkDetector`)
- Máquina de estados explícita com `EyeBlinkState` (`EYES_OPEN`, `POSSIBLE_CLOSING`, `CLOSURE_CONFIRMED`, `OPENING_CONFIRMED`, `GESTURE_COMPLETED`, `COOLDOWN`).
- Timestamps monotônicos substituindo contadores de frames.
- Histerese dupla (`ear_close_threshold` e `ear_open_threshold`) eliminando oscilações de borda.
- Calibração de referências individuais por olho (`left_open_ear`, `left_closed_ear`, `right_open_ear`, `right_closed_ear`).
- Discriminação fisiológica: micropiscada reflexa (<90ms e alta velocidade de abertura) vs. voluntária (120ms..450ms).

### 5.2 — Motor Central de Gestos e Arbitragem (`GestureEngine`)
- Arbitragem de conflitos bilaterais: janela de coincidência ($\le 80$ms) impede geração acidental de dois cliques.
- Descarte seguro de piscadas bilaterais naturais de lubrificação por padrão.
- **Click Freeze:** Buffer circular congelando o cursor na coordenada estável pré-fechamento ($\approx 120$ms antes da oclusão) durante a janela de clique ($150$ms), neutralizando a rotação do globo ocular (fenômeno de Bell) e deformações de landmarks.
- Supressão de cliques individuais acidentais durante arraste ativo.
- Remapeamento e desativação granular de gestos.

### 5.3 — Clique por Permanência (`DwellClicker`)
- Detecção temporal contínua de fixação ($R_{dwell} = 30$px, $T_{dwell} = 900$ms).
- Indicador de progresso normalizado ($0.0 \to 1.0$) para feedback visual circular.
- Cancelamento suave ao afastar o olhar ($> 1,35 \times R_{dwell}$).
- **Proteção Anti-Loop de Rearmamento:** Requer que o olhar se afaste $\ge 45$px antes de permitir novo clique no mesmo local.

### 5.4 — Barra de Ações Flutuante (`ActionBar`)
- Barra de ferramentas visual com botões dimensionados para acessibilidade (Esq, Dir, 2x, Arrastar/Soltar, Rolagem, Precisão, Pausa, Ancoragem).
- **Padrão Next-Action:** Armar ação para o próximo clique ou dwell, revertendo automaticamente para Clique Esquerdo.
- Ancoragem nos 4 cantos da tela (`TOP`, `BOTTOM`, `LEFT`, `RIGHT`).
- Acionamento por Dwell direto com consumo de evento (impede clique pass-through para o Windows).

### 5.5 — Modo Dedicado de Rolagem (`ScrollController`)
- Zonas direcionais de tela (topo 22% = rolar para cima; base 22% = rolar para baixo).
- Zona central neutra de 56% para leitura estável e confortável.
- Curva de velocidade proporcional à penetração na borda e rate limiting de 120ms.

### 5.6 — Perfis de Interação e Acessibilidade (`ProfileManager`)
- Perfis `CONTINUOUS`, `DWELL` e `HYBRID`.
- **Garantia de Acessibilidade:** Nenhuma função essencial exige piscada assimétrica.

### Critérios de Aceitação do Milestone 5
- [x] Detecção de piscada com máquina de estados explícita e temporização monotônica.
- [x] Click Freeze ativo impedindo saltos do cursor no clique.
- [x] Dwell click com proteção anti-loop e indicador de progresso.
- [x] Barra de ações com despacho Next-Action e ancoragem nos 4 lados.
- [x] Modo de rolagem dedicado com zona neutra de leitura.
- [x] Liberação garantida de arraste em pausas, encerramento ou perda de rosto > 1.0s.
- [x] Documentação técnica completa em `docs/GESTURE_ENGINE_INTERACTION.md`.
- [x] 302 testes unitários passando com 100% de sucesso (2.05s).

---

## Milestone 6 — Experiência de Uso, Acessibilidade e Funcionalidades

**Objetivo:** Transformar o EyeMouse de uma demonstração técnica em um protótipo utilitário acessível, compreensível, seguro e configurável.
**Dependencias:** Milestones 0 a 5 concluídos.
**Status:** CONCLUÍDO (2026-09-16)

### 6.1 — Redesenho do Painel de Controle (ControlPanel)
- Estrutura limpa em abas (`ttk.Notebook`): Visão Geral, Ajustes Rápidos, Avançado, Privacidade & Perfis.
- Badges e métricas em tempo real: estado da máquina de estados, FPS de captura/processamento, latência estimada de pipeline em milissegundos (~12-18 ms), selo de qualidade da calibração (<40px excelente, 40-80px boa, >80px imprecisa), estado de arraste e precisão.
- Navegação completa por teclado, botões de dimensões acessíveis e aviso legal explícito de tecnologia assistiva.

### 6.2 — Assistente de Configuração Inicial (SetupWizard)
- Fluxo guiado em 9 etapas: Boas-vindas $\to$ Seleção de Webcam $\to$ Posicionamento $\to$ Análise Fotométrica de Iluminação $\to$ Calibração do Olhar (16 pontos) $\to$ Calibração de Gestos $\to$ Perfil de Interação $\to$ Prática com Alvos $\to$ Ativação.
- Bloqueio estrito de avanço na calibração se inválida ou holdout error > 80px.
- Analisador fotométrico perceptual ITU-R 601 para diagnóstico da iluminação facial.

### 6.3 — Tela de Treinamento e Prática com Alvos (TrainingTargetUI)
- Ambiente de prática prévia com 3 tamanhos concêntricos calibrados: Grande (60px), Médio (40px) e Pequeno (25px).
- Coleta de métricas por tentativa: distância euclidiana ao centro (px), tempo de reação (ms), taxa de acerto acumulada (%) e cliques falsos.
- Relatório de desempenho multimodal e recomendações personalizadas ao término da sessão.

### 6.4 — Gerenciador de Configurações e Privacidade (SettingsManager)
- Persistência estruturada em JSON atômico com validação estrita de tipos e limites (`~/Documents/EyeMouse/profiles/`).
- Privacidade por design: processamento 100% local, zero persistência de imagens/vídeo/landmarks, e exclusão completa de dados a um clique.

### 6.5 — Documentação e Manual do Usuário
- Criação de `docs/USER_GUIDE.md` completo com ergonomia, iluminação, calibração, gestos, dwell, hotkeys e solução de problemas.
- Atualização do `README.md`.

### Critérios de Aceitação do Milestone 6
- [x] Painel de controle redesenhado e navegável por teclado.
- [x] Assistente de configuração inicial com análise fotométrica e portão de calibração rigoroso.
- [x] Tela de teste/treino com alvos de 3 diâmetros e relatório multimodal.
- [x] Persistência atômica de configurações e garantia estrita de privacidade local.
- [x] 345 testes unitários passando com 100% de sucesso (2.06s).

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
