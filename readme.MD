# Leitor automático de cédulas — Festival de Cinema de Bonito/MS

Lê as cédulas de votação (2 filmes por cédula, nota de 1 a 10 marcada em
quadradinhos) via webcam ou fotos, e extrai os votos automaticamente.

## Como o algoritmo funciona (visão geral)

1. **Localizar o papel na foto e corrigir a perspectiva** — a foto pode
   estar torta/em ângulo; o algoritmo acha os 4 cantos do papel e faz um
   "warp" pra obter uma visão de cima, sempre no mesmo tamanho.
2. **Binarizar** — separa "tinta" (marcações, texto impresso) do papel em
   branco, corrigindo variação de iluminação da própria foto.
3. **Ler o grid calibrado** — usa as posições exatas dos 20 quadradinhos
   (10 por filme), definidas uma vez em `grid_config.json`, e mede quanto
   de tinta tem dentro de cada um.
4. **Decidir a nota** — o quadradinho com mais tinta acima de um limiar
   vira a nota. Se nenhum ou mais de um quadradinho passar do limiar, a
   cédula é marcada como "revisar" em vez de arriscar um palpite errado.

Optei por **calibrar o grid uma única vez** (em vez de tentar "descobrir"
os quadradinhos automaticamente em cada imagem via contornos). É o jeito
como leitores de gabarito de prova funcionam de verdade: mais robusto a
sombra/textura do papel, e muito mais fácil de depurar quando alguma
cédula falhar, porque você sabe exatamente onde o sistema está olhando.

## Instalação

```bash
pip install -r requirements.txt
```

## Passo 1 — Calibrar (fazer uma vez só)

Tire uma foto de exemplo de uma cédula em branco (ou preenchida), nas
mesmas condições reais (mesma câmera, mesma mesa/suporte, mesma
iluminação que vai ser usada no dia do festival):

```bash
python calibrate.py exemplo_cedula.jpg
```

Siga as instruções na tela (cliques do mouse). No final, isso gera o
`grid_config.json`, que os outros dois scripts usam.

> **Dica pra confiabilidade:** fotografe sempre com o papel bem apoiado,
> ocupando a maior parte do quadro, sem sombra forte em cima dele. Se
> possível, fixe a câmera/webcam num suporte fixo (tipo suporte de
> celular) apontado pra uma mesa — assim a calibração vale pra todas as
> cédulas do evento inteiro, sem precisar recalibrar.

## Passo 2a — Processar fotos em lote

Se as cédulas forem fotografadas (celular, scanner) e salvas numa pasta:

```bash
python process_images.py pasta_com_fotos/
```

Gera `pasta_com_fotos/resultados/resultados.csv` com uma linha por
cédula (arquivo, status, nota do filme1, nota do filme2) e uma imagem de
depuração por cédula (útil pra conferir visualmente as que vieram como
"revisar").

## Passo 2b — Captura ao vivo pela webcam

Se preferir ficar passando as cédulas na frente da webcam durante a
apuração:

```bash
python webcam_capture.py
```

A tela mostra os quadradinhos detectados em tempo real (verde = marcado).
Quando a leitura estabilizar, aperte **ESPAÇO** pra confirmar e salvar o
voto em `resultados_webcam.csv`. Aperte **q** pra encerrar a sessão.

## Ajustando a sensibilidade

Se o sistema estiver marcando cédulas válidas como "revisar" com
frequência (ou aceitando marcações fracas demais), abra o
`grid_config.json` e ajuste `"fill_threshold"` (padrão 0.35):

- **Aumentar** (ex: 0.45) → fica mais rígido, exige um preenchimento mais
  "cheio" pra contar como voto.
- **Diminuir** (ex: 0.25) → aceita marcações mais fracas/parciais.

As imagens de depuração salvas por `process_images.py` mostram os
quadradinhos e ajudam a decidir esse ajuste olhando casos reais.

## Limitações conhecidas / próximos passos

- Funciona melhor se o papel tiver bordas retas e visíveis (nada de
  cédula muito amassada). Se a detecção automática do papel falhar com
  frequência (fundo bagunçado, pouco contraste), o `calibrate.py` permite
  marcar os 4 cantos manualmente como alternativa — o mesmo recurso pode
  virar um modo manual no `process_images.py` se for necessário.
- Rasuras (votante errou e cobriu o quadradinho errado) podem ser lidas
  como "múltiplas marcações" e cair em "revisar" — o que é o
  comportamento desejado (evita contar voto errado), mas exige checagem
  manual dessas cédulas no fim.
- Para os próximos festivais, se puderem redesenhar a cédula, colocar 4
  pequenos quadrados pretos sólidos nos cantos do papel (marcadores de
  referência) tornaria a detecção automática dos cantos ainda mais
  confiável em fundos bagunçados — mas não é necessário com o design
  atual.
