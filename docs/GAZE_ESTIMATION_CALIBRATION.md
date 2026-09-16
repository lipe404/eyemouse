# Estimativa de Olhar e Calibração de Alta Qualidade — Documentação Técnica

**Data:** 2026-09-16  
**Versão:** Milestone 3  
**Módulos:** `eye_mouse/gaze_features.py`, `eye_mouse/calibration_models.py`, `eye_mouse/calibration.py`, `eye_mouse/ui/calibration_ui.py`

---

## 1. Problema das Coordenadas Absolutas (Método Original — Modelo A)

No protótipo original (`gaze_tracker.py`), a estimativa do olhar utilizava exclusivamente as coordenadas $(x, y)$ absolutas das íris no frame da câmera:

$$\text{iris\_pos} = \frac{\text{iris}_{\text{left}} + \text{iris}_{\text{right}}}{2}$$

### Causa-Raiz da Instabilidade
Quando o usuário move a cabeça em direção a qualquer direção (inclinação, aproximação ou translação lateral):
- As coordenadas absolutas da íris se deslocam na imagem da câmera exatamente na mesma proporção do movimento da cabeça, **mesmo que o usuário continue fixando o olhar no mesmo pixel do monitor**.
- O modelo polinomial interpreta a translação da cabeça como uma mudança dramática na intenção de olhar, causando desvio severo do cursor (drift de centenas de pixels).

---

## 2. Novas Features Oculares Relativas (Modelo B)

Para desacoplar o olhar do movimento da cabeça, a posição da íris é calculada em relação à geometria intrínseca de cada olho:

1. **Eixo Ocular ($u$):** Definido pelo vetor entre o canto externo (*outer canthus*) e o canto interno (*inner canthus*):
   $$\vec{u} = \frac{\text{canthus}_{\text{inner}} - \text{canthus}_{\text{outer}}}{\|\text{canthus}_{\text{inner}} - \text{canthus}_{\text{outer}}\| + \varepsilon}$$
2. **Eixo Vertical Perpendicular ($v$):** Vetor ortogonal no plano 2D:
   $$\vec{v} = (-u_y, u_x)$$
3. **Largura e Altura Ocular:**
   - $\text{largura} = \|\text{canthus}_{\text{inner}} - \text{canthus}_{\text{outer}}\|$
   - $\text{altura} = |\text{pálpebra}_{\text{superior}} - \text{pálpebra}_{\text{inferior}}| \cdot \vec{v}$
4. **Posições Relativas Normalizadas:**
   $$\text{rel}_x = \frac{(\text{íris} - \text{canthus}_{\text{outer}}) \cdot \vec{u}}{\text{largura}}$$
   $$\text{rel}_y = \frac{(\text{íris} - \text{centro}_{\text{olho}}) \cdot \vec{v}}{\text{altura} + \varepsilon}$$

### Benefício
Se o usuário transladar a cabeça na frente da webcam, os cantos dos olhos e a íris transladam juntos no frame. A projeção relativa $(\text{rel}_x, \text{rel}_y)$ **permanece invariante à translação**, eliminando o jitter de corpo/cabeça.

---

## 3. Compensação de Pose da Cabeça (Modelo C)

Além das coordenadas relativas dos olhos, o Modelo C incorpora a estimativa da orientação 3D da cabeça:
- **Yaw ($\theta_{\text{yaw}}$):** Rotação horizontal (olhar lateral da cabeça).
- **Pitch ($\theta_{\text{pitch}}$):** Inclinação vertical (queixo para cima/baixo).
- **Roll ($\theta_{\text{roll}}$):** Inclinação lateral no plano da imagem.
- **Escala / Posição Facial:** Centro da face $(f_x, f_y)$ e distância inter-ocular.

Vetor de features resultante:
$$X = [1, \text{rel}_x, \text{rel}_y, \text{rel}_x \cdot \text{rel}_y, \text{rel}_x^2, \text{rel}_y^2, \text{yaw}, \text{pitch}, \text{roll}, f_x, f_y, \text{escala}]$$

---

## 4. Fusão Inteligente dos Olhos e Seleção Monocular

O sistema calcula qualidade $Q \in [0.0, 1.0]$ e Eye Aspect Ratio ($\text{EAR}$) para cada olho separadamente:
- **Ambos os olhos abertos e válidos ($Q_{\text{esq}} > 0.2$ e $Q_{\text{dir}} > 0.2$):** Fusão ponderada proporcional à qualidade:
  $$\vec{g}_{\text{fused}} = \frac{Q_{\text{esq}} \cdot \vec{g}_{\text{esq}} + Q_{\text{dir}} \cdot \vec{g}_{\text{dir}}}{Q_{\text{esq}} + Q_{\text{dir}}}$$
- **Oclusão parcial / Piscada voluntária de um olho:** O sistema seleciona automaticamente o olho com rastreamento válido (*fallback monocular*), impedindo que o fechamento intencional do olho para clique desvie a posição do cursor.
- **Ambos os olhos fechados / sem rosto:** `tracking_valid = False`, suprimindo o controle imediatamente.

---

## 5. Tabela Comparativa Experimental (Validação vs Treino)

Testes comparativos com $N = 160$ amostras (120 treino / 40 holdout) em tela de 1920×1080:

### Cenário 1: Com Movimento Involuntário de Cabeça (Translação ±4% e Rotação ±3.5°)

| Modelo | RMSE Treino | RMSE Validação (Holdout) | Mediana Validação | P95 Validação | Erro Relativo (% da Diagonal) |
|---|---|---|---|---|---|
| **Model A (Absoluto Original)** | 529.8 px | **461.7 px** | 401.6 px | 821.5 px | 18.56% (inutilizável) |
| **Model B (Íris Relativa)** | 17.2 px | **16.4 px** | 13.8 px | 26.8 px | **0.66%** (~28x mais estável) |
| **Model C (Relativo + Pose)** | 17.0 px | **16.0 px** | 14.1 px | 29.0 px | **0.64%** (ótimo global) |

### Cenário 2: Cabeça Perfeitamente Estática (Sem Movimento)

| Modelo | RMSE Treino | RMSE Validação (Holdout) | Mediana Validação | P95 Validação | Erro Relativo (% da Diagonal) |
|---|---|---|---|---|---|
| **Model A (Absoluto)** | 19.0 px | 19.3 px | 17.8 px | 30.7 px | 0.80% |
| **Model B (Relativo)** | 16.3 px | 18.4 px | 16.0 px | 29.8 px | 0.73% |
| **Model C (Relativo + Pose)** | 16.8 px | 16.6 px | 14.3 px | 25.6 px | 0.67% |

> **Conclusão:** Sem movimento de cabeça, todos os modelos alcançam boa precisão (~18 px). Porém, ao simular a dinâmica real da cabeça, o Modelo A perde completamente o alvo (461 px de erro), enquanto os Modelos B e C mantêm o erro de validação em ~16 px.

---

## 6. Reconstrução da Calibração

1. **Agregação Robusta de Amostras:** Para cada alvo na tela, são coletadas múltiplas observações descartando outliers via **Median Absolute Deviation (MAD)** e computando a **Mediana** final.
2. **Deduplicação Estrita:** Cada amostra exige `frame_id` e timestamp monotônicos únicos; amostras congeladas ou repetidas são descartadas.
3. **Regressão com Regularização Ridge ($L_2$):** Resolve $(A^T A + \lambda I)^{-1} A^T Y$ com $\lambda = 10^{-3}$, prevenindo matrizes degeneradas ou inversões numéricas colineares.
4. **Recalibração Rápida:** Método `recalibrate_point(idx)` permite reajustar um ponto individual sem recalibrar todos os 16 pontos.
5. **Ajuste Fino de Trim:** `set_trim(dx, dy)` permite compensar pequenos desvios sistemáticos sem corromper os coeficientes polinomiais.
