# EyeMouse — Guia Completo do Usuário e Manual de Acessibilidade

> **Aviso Legal & Médico**: O EyeMouse é um protótipo de software assistivo de código aberto destinado ao controle do cursor por meio de movimentos oculares e gestos faciais voluntários via webcam convencional. **Não é um dispositivo médico homologado** por agências regulatórias e não substitui diagnósticos ou terapias clínicas.

---

## 1. Requisitos do Sistema & Preparação do Ambiente

### 1.1 Iluminação e Câmera
- **Webcam**: Resolução mínima de 640x480 a 30 FPS ou 60 FPS (ex: Logitech C920, câmeras integradas HD).
- **Iluminação**: Utilize iluminação frontal suave e difusa (luz de teto ou luminária de mesa indireta).
  - ❌ **Evite contraluz**: Não sente de costas para janelas abertas ou lâmpadas fortes.
  - ❌ **Evite escuridão total**: A iluminação deve permitir que a íris e os cantos dos olhos sejam discerníveis sem ruído excessivo de sensor.
  - O **Assistente de Configuração** avalia a iluminação automaticamente por meio de fotometria perceptual (Luma média recomendada: 40 a 200).

### 1.2 Posicionamento e Ergonomia
- **Distância da Tela**: Mantenha-se entre **50 cm e 70 cm** de distância da webcam/monitor.
- **Altura do Monitor**: O topo do monitor deve ficar nivelado com a linha dos olhos.
- **Estabilidade da Cabeça**: Mantenha uma postura confortável e relaxada. Embora o EyeMouse compense pequenos movimentos de rotação e translação facial (yaw/pitch/roll), uma cabeça razoavelmente estável garante a melhor calibração.

---

## 2. Primeiro Uso: Assistente de Configuração Inicial (Setup Wizard)

Ao iniciar o EyeMouse ou clicar em **"Iniciar Assistente de Configuração"** no painel de controle, o assistente em 9 passos guia você pelo processo de configuração:

1. **Boas-Vindas & Termo Legal**: Explicação dos objetivos do sistema e termo de tecnologia assistiva.
2. **Seleção de Câmera**: Seleção da webcam correta e teste de captura em tempo real.
3. **Posicionamento**: Feedback visual para centralizar o rosto a 50-70 cm da tela.
4. **Verificação de Iluminação**: Análise fotométrica imediata (Escuro, Ideal ou Superexposto).
5. **Calibração do Olhar (16 pontos)**:
   - Fixe o olhar no círculo azul no centro de cada alvo até que ele seja registrado.
   - **Controle de Qualidade Estrito**: Se o erro de holdout for superior a **80px**, o sistema solicitará recalibração para evitar cursor impreciso ou saltitante.
6. **Calibração de Gestos (Opcional)**: Calibração de sensibilidade para detecção de piscadas voluntárias (EAR).
7. **Modo de Interação**: Escolha do perfil ideal para o seu perfil motor (Híbrido, Dwell ou Gestos).
8. **Prática com Alvos**: Treino preliminar na tela de prática antes de controlar o sistema operacional.
9. **Confirmação & Ativação**: Salva as preferências no perfil local e ativa o controle do mouse no Windows.

---

## 3. Modos e Perfis de Interação

O EyeMouse adapta-se às necessidades de diferentes usuários por meio de perfis configuráveis:

| Perfil | Descrição | Como Clicar | Ideal Para |
| :--- | :--- | :--- | :--- |
| **Híbrido (Padrão)** | Combina piscadas voluntárias para clique rápido com dwell de assistência. | Piscar olho esquerdo (clique esquerdo) ou fixar o olhar por 900 ms. | Maioria dos usuários; controle rápido e versátil. |
| **Dwell Only (Sem Piscadas)** | 100% livre de piscadas. Ideal para quem tem fadiga ocular ou espasmos involuntários. | Manter o olhar fixo em um ponto por **900 ms** dentro de um raio de 30px. | Usuários com mobilidade reduzida que não conseguem piscar voluntariamente. |
| **Contínuo (Gestos)** | Acionamento exclusivo por piscadas e gestos bilaterais voluntários. | Piscada rápida voluntária esquerda/direita; piscar prolongado para arrastar. | Usuários acostumados a gestos faciais ágeis. |

---

## 4. Comandos e Gestos Suportados

### 4.1 Movimentação do Cursor
- O cursor acompanha o centro do seu olhar na tela de forma suave, utilizando o **One Euro Filter** adaptativo (filtra tremores microscópicos durante fixações, respondendo instantaneamente a sacadas rápidas).

### 4.2 Ações do Mouse
- **Clique Esquerdo**:
  - Piscada voluntária do olho esquerdo (66 ms a 500 ms), OU
  - Fixação do olhar (Dwell) por 900 ms sobre o alvo.
- **Clique Direito**:
  - Piscada voluntária do olho direito (66 ms a 500 ms), OU
  - Selecionar "Clique Direito" na Barra de Ações e fixar o olhar no elemento desejado.
- **Duplo Clique**:
  - Duas piscadas voluntárias consecutivas do olho esquerdo dentro de 400 ms, OU
  - Selecionar "Duplo Clique" na Barra de Ações.
- **Arrastar e Soltar (Drag & Drop)**:
  - Piscar e segurar qualquer olho por mais de **1.2 segundos** (inicia arraste), mover o cursor e piscar novamente para soltar, OU
  - Clicar no botão "Arrastar" na Barra de Ações.
- **Modo Precisão (Velocidade Reduzida)**:
  - Reduz a sensibilidade do cursor para 35% em torno da área de interesse, facilitando selecionar links pequenos ou botões de menu.
  - Acionável pela Barra de Ações ou tecla de atalho.
- **Modo Rolagem (Scroll)**:
  - Ao ativar a rolagem na Barra de Ações, olhar para o terço superior da tela rola a página para cima; olhar para o terço inferior rola para baixo. A faixa central atua como zona neutra de leitura estável.

---

## 5. Tecla de Atalho Global de Emergência

- **`Ctrl + Shift + P`**:
  - **Pausa Imediata**: Congela o cursor e desativa a injeção de cliques.
  - **Liberação de Segurança (Safety Release)**: Qualquer botão do mouse mantido pressionado é imediatamente liberado pelo sistema operacional.
  - **Retomada**: Pressione novamente `Ctrl + Shift + P` para retomar o controle ocular.

---

## 6. Tela de Treinamento e Prática com Alvos

Para praticar antes de operar aplicativos do Windows, utilize a **Tela de Treino**:
- Apresenta alvos circulares concêntricos em 3 dimensões calibradas:
  - **Grande (60px)**: Adaptação inicial e controle global.
  - **Médio (40px)**: Tamanho padrão de botões e caixas de diálogo do Windows.
  - **Pequeno (25px)**: Equivalente a ícones da bandeja do sistema e links web.
- **Métricas Apresentadas**:
  - Taxa de acerto acumulada (%).
  - Distância média em pixels em relação ao centro do alvo.
  - Tempo médio de reação por alvo (segundos).
  - Contagem de cliques falsos ou dispersos.
- **Feedback Acessível**: Mensagens textuais e ícones gráficos claros (`[✓ ACERTOU]` / `[✗ FORA]`), sem depender de cores para acessibilidade a daltônicos.

---

## 7. Painel de Controle (Control Panel)

O painel de controle apresenta 4 abas organizadas:

1. **Visão Geral**:
   - Estado em tempo real (Ativo, Pausado, Rosto Não Detectado).
   - FPS de captura da câmera e de processamento de rede neural.
   - Latência estimada de ponta a ponta em milissegundos (~12-18 ms).
   - Selo de qualidade da calibração:
     - **Excelente (< 40px)**: Precisão milimétrica.
     - **Boa (40-80px)**: Adequada para a maioria das tarefas.
     - **Imprecisa (> 80px)**: Recomendado recalibrar.
   - Indicadores de arraste e modo precisão.
2. **Ajustes Rápidos**:
   - Seletor imediato de perfil de interação.
   - Controle deslizante de estabilidade / suavização.
   - Alternadores rápidos de modo de precisão e barra de ações.
3. **Avançado**:
   - Ajuste de parâmetros de Dwell Click e anti-loop.
   - Recalibração de sensibilidade de pálpebras (EAR) com histerese dupla.
4. **Privacidade & Perfis**:
   - Acesso ao Assistente de Configuração.
   - Botão para **Apagar Dados e Calibrações** permanentemente.
   - Garantias de privacidade local.

---

## 8. Privacidade dos Dados

- **Processamento 100% Local**: Todo o pipeline de rastreamento facial e processamento de imagem roda na CPU local do computador.
- **Zero Transmissão**: Nenhuma imagem da webcam, frame de vídeo ou coordenada biométrica é enviada para a internet.
- **Persistência Mínima**: Apenas números e opções de configuração (sensibilidade, coeficientes de polinômio da calibração) são salvos em `~/Documents/EyeMouse/profiles/`.
- **Exclusão Total**: O usuário pode a qualquer momento clicar em "Apagar Meus Dados" para remover todos os arquivos de configuração locais.

---

## 9. Solução de Problemas (Troubleshooting)

| Problema | Possível Causa | Solução Recomendada |
| :--- | :--- | :--- |
| **Cursor treme durante fixações** | Filtro de suavização muito responsivo | Aumente ligeiramente a suavização na aba "Ajustes Rápidos" do painel. |
| **Cursor atrasado / lag percebido** | FPS da câmera baixo ou CPU sobrecarregada | Verifique se a câmera está a 30 FPS. Feche processos pesados em segundo plano. |
| **Clique involuntário ao piscar** | Limiar de piscada (EAR) muito alto | Execute a "Calibração de Sensibilidade de Piscada" ou mude para o perfil "Dwell Only". |
| **Cursor não alcança cantos da tela** | Calibração com pouca amplitude ocular | Recalibre olhando bem próximo aos alvos dos extremos da tela durante o wizard. |
| **"Rosto Não Detectado" intermitente** | Iluminação fraca ou reflexo em óculos | Posicione uma luminária indireta à sua frente; incline levemente os óculos para reduzir reflexos diretos. |
| **Cursor travou clicando/arrastando** | Interrupção durante clique | Pressione `Ctrl + Shift + P` para acionar a liberação de emergência (Safety Release). |
