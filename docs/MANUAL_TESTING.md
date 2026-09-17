# Roteiro de Testes Manuais de Validação — EyeMouse Release Candidate (Milestone 7)

Este documento contém o protocolo estrito de validação manual para homologação do EyeMouse no ambiente operacional Windows 10/11.

---

## Matriz de Cenários de Teste

| ID | Cenário | Objetivo | Procedimento | Comportamento Esperado | Status |
|:--:|---|---|---|---|:---:|
| **TM-01** | Movimentação em Toda a Tela | Verificar se o cursor atinge os 4 cantos da área de trabalho ($0$ a $W$, $0$ a $H$). | Olhar sequencialmente para os cantos superior esquerdo, superior direito, inferior esquerdo e inferior direito. | O cursor deve atingir suavemente os quatro cantos sem travar ou sofrer corte artificial de coordenadas. | [ ] Aprovado |
| **TM-02** | Seleção de Botões Pequenos | Testar a precisão fina e estabilidade do One Euro Filter. | Na tela de treino (`TrainingTargetUI`), selecionar os alvos pequenos de 25 px ou tentar clicar no ícone de fechar uma janela. | O cursor deve desacelerar e estabilizar sem jitter (tremor < 1.5 px), permitindo acerto com o perfil Dwell ou piscada. | [ ] Aprovado |
| **TM-03** | Execução de Duplo Clique | Validar acionamento atômico de duplo clique sem deslocar o cursor. | No perfil Híbrido, usar a Barra de Ações (botão Duplo Clique) ou piscada rápida sequencial sobre uma palavra ou pasta. | Ação de duplo clique deve abrir a pasta ou selecionar a palavra inteira sem mover o ponteiro entre cliques. | [ ] Aprovado |
| **TM-04** | Abertura de Menu de Contexto | Validar clique com botão direito. | Selecionar a ação "Direito" na Barra de Ações ou executar piscada com o olho direito por ~150 ms. | O menu de contexto do Windows abre imediatamente sob a posição pré-oclusão do olhar. | [ ] Aprovado |
| **TM-05** | Arraste de Janela / Item | Testar início, movimentação e término do modo de arraste (Drag & Drop). | Fechar ambos os olhos por 1.5s ou acionar o botão "Arrastar" na Barra de Ações. Mover o olhar para a nova posição e piscar/acionar para soltar. | A janela é mantida presa ao cursor durante todo o deslocamento e solta exatamente na coordenada final. | [ ] Aprovado |
| **TM-06** | Seleção Contínua de Texto | Validar estabilidade de arraste em precisão microscópica. | Iniciar arraste no início de um parágrafo em um editor de texto ou navegador, mover o olhar até o fim do texto e soltar. | O bloco de texto desejado é selecionado sem cancelamentos prematuros. | [ ] Aprovado |
| **TM-07** | Rolagem Direcional de Página | Validar modo Scroll dedicado. | Ativar o modo Scroll na Barra de Ações (`[Rolagem]`). Fixar o olhar no topo (22%) ou base (22%) da tela. | A página rola suavemente em pulsos de 120 ms para cima ou para baixo. Olhar para o centro interrompe o scroll. | [ ] Aprovado |
| **TM-08** | Navegação em Bordas de Tela | Testar comportamento nas extremidades do monitor. | Mover o cursor até a barra de tarefas inferior e bordas laterais. | Não há estouro de array, congelamento de ponteiro ou oscilação infinita contra a borda. | [ ] Aprovado |
| **TM-09** | Pausa e Retomada Global | Testar o hotkey de emergência `Ctrl+Shift+P`. | Pressionar `Ctrl+Shift+P` enquanto o cursor estiver em movimento. Mover os olhos. Pressionar `Ctrl+Shift+P` novamente. | Em pausa, o cursor congela e botões são soltos. Na retomada, o filtro reinicia suavemente sem saltos. | [ ] Aprovado |
| **TM-10** | Desconexão Física da Webcam | Avaliar robustez contra perda de hardware de captura. | Durante o rastreamento ativo, desconectar o cabo USB da webcam. | O pipeline detecta a falha, transiciona para `AppState.ERROR`, desarma botões (`release_all`) e exibe aviso amigável. | [ ] Aprovado |
| **TM-11** | Encerramento Durante Arraste | Garantir que botões do mouse não fiquem presos no Windows. | Iniciar um arraste de arquivo. Com o botão pressionado, fechar o EyeMouse via `Alt+F4` ou no gerenciador de tarefas. | O gancho `atexit` e o `quit_app()` emitem `LEFTUP`. O mouse físico volta a operar normalmente sem cliques presos. | [ ] Aprovado |
| **TM-12** | Alternância de Resolução de Tela | Testar mudança dinâmica de resolução (ex: 1080p para 720p ou 1440p). | Com a aplicação calibrada, mudar a resolução do monitor nas Configurações de Vídeo do Windows. | O `CalibrationManager` adapta os fatores de escala `_scale_x` e `_scale_y` proporcionalmente sem quebrar o cursor. | [ ] Aprovado |
| **TM-13** | Escala de DPI (100%, 125%, 150%) | Testar a conscientização Per-Monitor DPI V2. | No Windows, alterar a escala para 125% ou 150%. Mover o cursor para os extremos. | As coordenadas SendInput continuam alinhadas com os pixels reais sem desvio de escala do DWM. | [ ] Aprovado |
| **TM-14** | Múltiplos Monitores (Virtual Desktop) | Avaliar suporte a tela estendida. | Em setup com 2 monitores, arrastar o cursor para o monitor secundário. | O cursor transita livremente entre monitores usando coordenadas absolutas no desktop virtual (`MOUSEEVENTF_VIRTUALDESK`). | [ ] Aprovado |

---

## Testes de Recuperação Ambiental

### R-01: Mudança Brusca de Iluminação
- **Procedimento**: Durante o rastreamento, apagar a luz do cômodo ou acender uma lâmpada lateral forte.
- **Resultado Esperado**:
  - Se a iluminação cair abaixo do limiar crítico (< 40 na escala ITU-R 601), a máquina de estados entra em `TRACKING_LOST`.
  - O cursor para de se movimentar (evitando saltos aleatórios por sombras).
  - Ao normalizar a iluminação, o rastreamento recupera em menos de 300 ms após os 5 frames de estabilização.

### R-02: Movimento Brusco da Cabeça (Deslocamento Físico)
- **Procedimento**: Levantar da cadeira por 5 segundos e retornar à posição de uso (50 a 70 cm da tela).
- **Resultado Esperado**:
  - `TRACKING_LOST` ativado de forma segura.
  - Ao retornar, a validação temporal descarta frames antigos e o `OneEuroFilter.reset()` impede o cursor de saltar da posição antiga.
