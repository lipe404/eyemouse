# Pipeline de Visão Computacional de Baixa Latência — Documentação Técnica

**Data:** 2026-09-16  
**Versão:** Milestone 2  
**Módulos:** `eye_mouse/camera_capture.py`, `eye_mouse/gaze_tracker.py`, `eye_mouse/tracking_validator.py`, `eye_mouse/frame_data.py`

---

## 1. Backends de Captura OpenCV no Windows

O OpenCV no Windows suporta diferentes camadas de abstração para acesso a webcams UVC (USB Video Class):

| Backend | Constante OpenCV | Vantagens | Desvantagens | Recomendação |
|---|---|---|---|---|
| **DirectShow** | `cv2.CAP_DSHOW` | • Negociação rápida de formatos.<br>• Suporta `CAP_PROP_FOURCC = 'MJPG'`.<br>• Menor latência de driver na maioria dos sensores USB.<br>• Respeita buffers menores. | • API mais antiga do Windows. | **Padrão Recomendado** (`CAMERA_BACKEND = "DSHOW"`) |
| **Media Foundation** | `cv2.CAP_MSMF` | • API nativa moderna da Microsoft.<br>• Suporta resoluções 4K e novos formatos HDR/YUV. | • Buffer interno opaco do Windows (frequentemente ignora `CAP_PROP_BUFFERSIZE`).<br>• Pode introduzir 2 a 3 frames de latência em fila oculta. | Alternativa de compatibilidade |
| **Default / Any** | `cv2.CAP_ANY` | • Fallback universal automático. | • Não previsível qual camada será selecionada. | Apenas fallback |

### Tratação Defensiva de Parâmetros
O `CameraCapture` audita os parâmetros solicitados contra os valores efetivamente aceitos pela câmera via `cap.get()`. Se a câmera rejeitar resolução, FourCC ou `CAP_PROP_BUFFERSIZE`, o evento é registrado em `camera_metadata["rejected_configs"]` sem interrupção do sistema.

---

## 2. Comparativo de Resoluções de Captura

A resolução do frame afeta diretamente a precisão do centróide da íris e o tempo de inferência do MediaPipe:

| Resolução | Tamanho da Íris na Imagem | Custo de Inferência (CPU) | Estabilidade do Cursor | FPS Típico | Veredito |
|---|---|---|---|---|---|
| **320×240** | ~8 a 12 pixels | ~8–12 ms | Baixa (erro de quantização elevado; cursor com micro-saltos) | 30–60 FPS | Não recomendado para uso principal |
| **640×480** | ~20 a 28 pixels | ~12–18 ms | **Alta** (sub-pixel preciso; íris nítida) | 30 FPS | **Ideal (Equilíbrio Precisão/Latência)** |
| **1280×720** | ~40 a 55 pixels | ~35–55 ms | Muito Alta | 15–22 FPS (queda de fluidez) | Apenas para CPUs de alto desempenho |

> **Conclusão:** A redução forçada para 320×240 que existia no protótipo original degradava a precisão da íris pela metade. A versão 2.0 utiliza 640×480 como resolução padrão de processamento.

---

## 3. Modos do MediaPipe Face Landmarker

| Modo | Método | Timestamps | Filtro Temporal Interno | Comportamento |
|---|---|---|---|---|
| **`RunningMode.VIDEO`** | `detect_for_video(image, ts_ms)` | Monotônicos estritamente crescentes (`ms`) | **Sim** (Kalman/OneEuro interno entre frames) | **Padrão:** Combina estabilidade temporal nativa do MediaPipe com execução síncrona determinística na thread de processamento. |
| **`RunningMode.LIVE_STREAM`** | `detect_async(image, ts_ms)` | Monotônicos estritamente crescentes (`ms`) | **Sim** | **Experimental:** Não bloqueante via callback, mas introduz risco de reordenamento de frames sob picos de carga. |
| **`RunningMode.IMAGE`** | `detect(image)` | Não aplicável | **Não** (trata cada frame isoladamente) | Maior ruído e jitter de coordenadas. |

### Otimizações de Sobrecarga (Blendshapes e Transformações)
No Milestone 2, `output_face_blendshapes` e `output_facial_transformation_matrixes` foram tornados configuráveis via `config.py` e desativados por padrão (`MEDIAPIPE_BLENDSHAPES = False`), reduzindo o tempo de inferência em aproximadamente **15% a 20%**.

---

## 4. Arquitetura Latest-Frame e Validação Temporal

```
Webcam
  │
  ▼
CameraCaptureThread (buffer mínimo de 1 slot)
  │  [Sobrescreve frames se a thread de processamento estiver ocupada]
  ▼
FramePacket (frame_id, capture_timestamp, image, camera_metadata)
  │
  ▼
ProcessingThread (GazeTracker RunningMode.VIDEO)
  │
  ▼
TrackingResult (landmarks, features, tracking_valid, tracking_quality)
  │
  ▼
TrackingValidator:
  ├── 1. Checagem de ID estritamente crescente
  ├── 2. Checagem de expiração: idade > MAX_OBSERVATION_AGE_SEC (150ms) -> REJEITADO
  └── 3. Período de estabilização pós-recuperação de rosto (5 frames / 200ms)
        ├── Se estabilizando -> suprime cliques e movimentos de mouse
        └── Se estabilizado  -> envia para MouseController
```

### Regra Crítica de Segurança
- **Frames com idade > 150 ms**: O cursor não se move e nenhum gesto é processado.
- **Rosto recuperado após perda**: O cursor só retoma ação após `TRACKING_STABILIZATION_FRAMES = 5` frames contínuos válidos, impedindo falsos cliques decorrentes do reposicionamento inicial do rosto.

---

## 5. Visualização e Debug Desacoplados

- **Preview Opcional**: `SHOW_PREVIEW = True/False`. Em segundo plano, o consumo de recursos cai consideravelmente sem preview de tela.
- **Taxa de Renderização Independente**: O preview visual é renderizado em `PREVIEW_FPS = 15` FPS, enquanto o rastreamento roda na taxa total da câmera (30 FPS), economizando ciclos de CPU no desenho do Tkinter.
- **Desenho Otimizado**: Landmarks e polígonos dos olhos só são gerados se `DEBUG_DRAW = True`.
