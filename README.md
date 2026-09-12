# Leitor automático de cédulas

Lê as cédulas de votação (2 filmes por cédula, nota de 1 a 10 marcada em
quadradinhos) via webcam ou fotos, e extrai os votos automaticamente.

## O que mudou nesta versão (testado com fotos reais, de propósito ruins)

- **Localização do papel pela região clara, não pela borda.** Mesa de
  madeira com veio escuro confundia tanto detecção de borda (Canny)
  quanto busca de blobs escuros — mas o papel continua sendo, disparado,
  a maior região CLARA e uniforme da cena. Agora o sistema acha o papel
  assim, e só depois ajusta um retângulo (`minAreaRect`) em volta dele.
  Isso se provou robusto até com a mão tampando parte da borda, mesa
  bagunçada com outros objetos no fundo, e fotos malfeitas em geral.
- **Ajuste fino das linhas divisórias.** Quadradinhos desenhados à mão
  quase nunca têm exatamente o mesmo tamanho — a primeira caixa de uma
  coluna pode ser visivelmente maior que a última. Dividir o espaço
  calibrado em partes IGUAIS acumula esse erro e desalinha as últimas
  caixas. Agora, a cada divisória, o sistema parte do palpite de
  espaçamento igual mas ajusta pra posição real da linha de tinta mais
  próxima (busca sequencial: cada divisória usa a ANTERIOR como
  referência, então o erro não se acumula ao longo da fileira/coluna).
- **Duas margens em vez de uma.** As bordas laterais de uma coluna (ou
  topo/base de uma fileira) são UMA linha contínua que atravessa TODAS as
  caixas — diferente da linha divisória, que é por-caixa. Se a margem de
  leitura for pequena demais nessa direção, toda caixa herda um "chão" de
  tinta vindo dessa borda, o que atrapalha a comparação relativa entre
  elas. Agora existem duas margens independentes: uma pequena pra excluir
  a divisória entre caixas, e uma maior pra excluir a borda lateral
  contínua.

Com essas mudanças, todas as fotos reais de teste (incluindo as bem
escuras, com mesa de madeira no fundo, mão tampando borda, e marcação
fina em X ou risco) passaram a ser lidas corretamente — com uma exceção
documentada abaixo.

## Limitação conhecida (vale a pena entender)

Em desenho muito à mão livre, a linha lateral de uma coluna pode "bambear"
pra dentro em algum ponto (não é reta, curva um pouco). Quando isso
acontece bem perto de uma das margens de leitura, o sistema pode contar
aquele trecho como se fosse marcação, criando falso empate com a caixa
realmente marcada — e a cédula cai em "revisar" em vez de decidir sozinha.
Isso é o comportamento CORRETO diante da incerteza (prefere pedir
conferência manual a arriscar um palpite errado), mas vale saber que
cédulas desenhadas às pressas, com linhas bem tortas, podem gerar mais
"revisar" do que o necessário. Em uma cédula impressa (linhas retas de
verdade) esse efeito praticamente desaparece.

## Como o algoritmo funciona (visão geral)

1. **Localizar o papel e corrigir a perspectiva** — acha a maior região
   clara da foto (o papel), ajusta um retângulo em volta dela, e aplica
   a correção de perspectiva pra obter uma imagem "vista de cima", sempre
   no mesmo tamanho. Se houver marcadores de canto (4 quadradinhos pretos
   desenhados perto das quinas do papel), o sistema tenta usá-los primeiro
   pra um resultado ainda mais preciso; senão, usa só o formato do papel.
2. **Binarizar** — separa "tinta" (marcações, texto impresso) do papel em
   branco, corrigindo variação de iluminação da própria foto.
3. **Ajustar o grid à tinta real** — usa as posições aproximadas
   calibradas em `grid_config.json` como ponto de partida, mas refina cada
   linha divisória pra onde a tinta realmente está.
4. **Decidir a nota de forma relativa** — o quadradinho com claramente
   mais tinta que os outros da mesma fileira/coluna vence (funciona pra
   preenchimento completo, X, ou risco). Se nenhum se destacar, ou dois
   ficarem próximos demais, a cédula vai pra "revisar" em vez de arriscar.

## Instalação

```bash
pip install -r requirements.txt
```

## Passo 1 — Calibrar (fazer uma vez só)

```bash
python calibrate.py exemplo_cedula.jpg
```

Siga as instruções na tela (cliques do mouse). Gera o `grid_config.json`
usado pelos outros dois scripts.

## Passo 2a — Processar fotos em lote

```bash
python process_images.py pasta_com_fotos/
```

Gera `resultados.csv` (arquivo, status, método de detecção, nota de cada
filme) e uma imagem de depuração por cédula.

## Passo 2b — Captura ao vivo pela webcam

```bash
python webcam_capture.py
```

Verde = marcado com confiança. Aperte **ESPAÇO** pra confirmar e salvar
o voto, **q** pra sair.

## Ajustando a sensibilidade

No `grid_config.json`:

- `"min_ink_ratio"` (padrão 0.02) — tinta mínima pra um quadradinho ser
  candidato a marcado. Diminua se marcações fracas estão virando
  "sem_marcacao"; aumente se ruído está sendo confundido com marcação.
- `"margin_factor"` (padrão 1.6) — quanto o quadradinho líder precisa se
  destacar do segundo colocado. Diminua (ex: 1.3) se cédulas válidas
  caem em "ambiguo" com frequência.
- `"margin_ratio"` de cada grupo (padrão 0.12) — margem que exclui a
  linha divisória ENTRE caixas.
- `"margin_ratio_cross"` de cada grupo (padrão 0.24) — margem que exclui
  a borda lateral CONTÍNUA (a que atravessa todas as caixas de uma vez).
  Aumente se essa borda estiver vazando pra dentro das leituras; mas
  lembre que uma linha muito torta pode não ter solução só com esse
  ajuste (ver limitação acima).

As imagens de depuração do `process_images.py` mostram, caixa por caixa,
qual "ganhou" — ajudam muito a calibrar esses números olhando casos reais.

## Limitações conhecidas / próximos passos

- Depende de o papel ser a região mais clara e uniforme da cena (verdade
  na esmagadora maioria das fotos de documento em cima de mesa).
- Rasuras tendem a virar "ambiguo" e cair em "revisar" — proposital.
- Manter os marcadores de canto ajuda na precisão, mas não é mais
  estritamente necessário — o método de contorno sozinho já se mostrou
  robusto nos testes reais.
