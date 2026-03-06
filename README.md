# EyeMouse - Controle de Mouse por Rastreamento Ocular

O EyeMouse é uma aplicação de tecnologia assistiva desenvolvida em Python que permite controlar o cursor do mouse utilizando apenas o movimento dos olhos e executar cliques através de piscadas. O sistema utiliza visão computacional e aprendizado de máquina para mapear a posição da íris na tela em tempo real.

## Funcionalidades Principais

*   **Rastreamento Ocular em Tempo Real**: Utiliza a webcam para detectar e seguir o movimento da íris com alta precisão.
*   **Calibração Personalizada**: Sistema de calibração de 9 pontos para mapear as características únicas do olhar de cada usuário para as coordenadas da tela.
*   **Cliques por Piscada**: Detecção de piscadas voluntárias para executar cliques esquerdo, direito e duplo clique.
*   **Modo Arrastar**: Funcionalidade de "segurar e arrastar" ativada ao fechar o olho por um período prolongado.
*   **Suavização de Movimento**: Algoritmos de filtro para garantir um movimento de cursor fluido e reduzir tremores naturais.
*   **Painel de Controle**: Interface flutuante para ajustes de sensibilidade, recalibração e monitoramento do estado do sistema.

## Gestos e Comandos

O sistema utiliza os seguintes padrões de piscada para interagir com o computador:

| Gesto | Ação | Descrição |
|---|---|---|
| **Piscar Olho Esquerdo** | Clique Esquerdo | Uma piscada rápida e intencional com o olho esquerdo. |
| **Piscar Olho Direito** | Clique Direito | Uma piscada rápida e intencional com o olho direito. |
| **Piscar Ambos** | Duplo Clique | Piscar ambos os olhos simultaneamente de forma rápida. |
| **Fechar Olho Esq. (1.5s)** | Segurar / Arrastar | Manter o olho esquerdo fechado por 1,5 segundos ativa o modo "arrastar". Fechar novamente solta o botão. |

**Nota**: O sistema diferencia piscadas naturais (involuntárias) de piscadas de comando baseando-se na duração e intensidade do fechamento da pálpebra.

## Como Funciona

### Rastreamento e Calibração
O software utiliza o MediaPipe Face Mesh para identificar 468 pontos de referência facial. A partir desses pontos, isola-se a região dos olhos e calcula-se o centro da íris.
Durante a calibração, o sistema coleta amostras da posição da íris enquanto o usuário olha para pontos fixos na tela. Um modelo de regressão polinomial é então treinado para correlacionar a geometria do olho com as coordenadas `(x, y)` do monitor.

### Detecção de Piscadas (EAR)
A detecção de cliques baseia-se no cálculo do EAR (Eye Aspect Ratio), uma medida matemática da abertura do olho. Quando o EAR cai abaixo de um limiar definido (configurável), o sistema registra um fechamento. Filtros temporais são aplicados para distinguir piscadas reais de ruídos ou piscadas reflexas muito rápidas.

## Stack Tecnológico e Dependências

O projeto foi desenvolvido inteiramente em **Python 3.10+** utilizando as seguintes bibliotecas:

*   **MediaPipe**: Framework do Google para processamento de mídia e detecção de landmarks faciais de alta performance via CPU.
*   **OpenCV (cv2)**: Captura de vídeo da webcam e pré-processamento de imagem.
*   **NumPy**: Computação numérica para cálculos vetoriais, médias ponderadas e modelos de regressão para calibração.
*   **PyAutoGUI**: Automação de interface para controle programático do cursor e eventos de clique do sistema operacional.
*   **Tkinter**: Interface gráfica nativa do Python utilizada para a tela de calibração fullscreen e o painel de controle.

## Instalação e Execução

1.  Certifique-se de ter o Python 3.10 ou superior instalado.
2.  Instale as dependências listadas:
    ```bash
    pip install -r eye_mouse/requirements.txt
    ```
3.  Execute o arquivo principal dentro do diretório do projeto:
    ```bash
    cd eye_mouse
    python main.py
    ```

## Configuração

O arquivo `config.py` permite ajustar parâmetros avançados para adaptar o sistema ao seu ambiente e preferências:

*   **CAMERA_INDEX**: Índice da webcam a ser utilizada.
*   **EMA_ALPHA**: Fator de suavização do cursor (valores menores = mais suave, porém com maior latência).
*   **BLINK_EAR_THRESHOLD**: Sensibilidade da detecção de piscada (ajuste se os cliques não estiverem sendo registrados ou ocorrendo involuntariamente).
