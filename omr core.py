"""
omr_core.py
------------
Núcleo do algoritmo de leitura das cédulas do festival de Bonito.

Pipeline:
  1. Encontrar o contorno do papel na imagem (foto de webcam/celular) e
     corrigir a perspectiva -> imagem "vista de cima", em tamanho padrão.
  2. Binarizar (transformar em preto/branco) para separar tinta de fundo.
  3. Usar as coordenadas de grid_config.json (geradas por calibrate.py) para
     recortar cada um dos 20 quadradinhos (10 por filme x 2 filmes).
  4. Medir a proporção de "tinta" dentro de cada quadradinho -> decidir qual
     foi marcado.

Este módulo não depende de webcam nem de arquivos em disco: recebe imagens
já carregadas (arrays numpy/OpenCV) e devolve resultados estruturados.
"""

import json
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Localização do papel e correção de perspectiva
# ---------------------------------------------------------------------------

def order_points(pts):
    """Ordena 4 pontos como [topo-esq, topo-dir, baixo-dir, baixo-esq]."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]        # topo-esquerdo: menor soma x+y
    ordered[2] = pts[np.argmax(s)]        # baixo-direito: maior soma x+y
    ordered[1] = pts[np.argmin(diff)]     # topo-direito: menor (y-x)
    ordered[3] = pts[np.argmax(diff)]     # baixo-esquerdo: maior (y-x)
    return ordered


def find_document_contour(image, min_area_ratio=0.2):
    """
    Tenta achar automaticamente o contorno de 4 lados (o papel) na imagem.
    Retorna os 4 pontos (não ordenados) ou None se não encontrar.

    min_area_ratio: o contorno candidato precisa ocupar pelo menos essa
    fração da área total da imagem, pra não confundir com objetos pequenos
    no fundo (ex: um crachá em cima da mesa).
    """
    h, w = image.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * min_area_ratio:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def four_point_transform(image, pts, canvas_size):
    """Aplica warpPerspective para 'endireitar' o papel num canvas fixo."""
    rect = order_points(pts)
    W, H = canvas_size
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (W, H))
    return warped


def warp_document(image, canvas_size=(1000, 700), manual_corners=None):
    """
    Retorna a imagem do papel 'endireitada' em canvas_size, ou None se não
    conseguir localizar o papel automaticamente e nenhum canto manual foi
    passado.

    manual_corners: lista de 4 pontos (x, y) opcional, usada quando a
    detecção automática falha (ex: fundo bagunçado) — normalmente vinda de
    cliques manuais no calibrate.py.
    """
    if manual_corners is not None:
        return four_point_transform(image, manual_corners, canvas_size)

    corners = find_document_contour(image)
    if corners is None:
        return None
    return four_point_transform(image, corners, canvas_size)


# ---------------------------------------------------------------------------
# 2. Binarização
# ---------------------------------------------------------------------------

def binarize(warped_bgr):
    """
    Converte a imagem já corrigida em uma imagem binária onde tinta/marca
    = branco (255) e papel em branco = preto (0).

    Importante: threshold adaptativo local (janela pequena) parece uma boa
    ideia pra tolerar iluminação desigual, mas quebra exatamente no caso que
    mais importa aqui — um quadradinho todo preenchido de caneta, onde o
    "fundo local" dentro da própria mancha de tinta já é escuro, então a
    marcação inteira some. Em vez disso: primeiro estimamos e removemos o
    gradiente de iluminação (dividindo pela versão bem desfocada da imagem,
    técnica de "flat-fielding"), e só depois aplicamos um threshold global
    (Otsu) — isso lida bem tanto com sombra/luz desigual na foto quanto com
    áreas grandes e uniformemente preenchidas.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    _, binary = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


# ---------------------------------------------------------------------------
# 3. Grid calibrado -> células dos quadradinhos
# ---------------------------------------------------------------------------

def load_config(path="grid_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_row_cells(row_cfg):
    """
    A partir de um bloco do config (uma fileira = um filme), calcula as
    coordenadas (x1,y1,x2,y2) de cada um dos n_boxes quadradinhos,
    distribuídos uniformemente entre x_start e x_end.
    """
    x_start, x_end = row_cfg["x_start"], row_cfg["x_end"]
    y_start, y_end = row_cfg["y_start"], row_cfg["y_end"]
    n = row_cfg["n_boxes"]
    cell_w = (x_end - x_start) / n

    cells = []
    for i in range(n):
        x1 = int(round(x_start + i * cell_w))
        x2 = int(round(x_start + (i + 1) * cell_w))
        cells.append((x1, y_start, x2, y_end))
    return cells


def cell_fill_ratio(binary_img, cell, margin_ratio=0.18):
    """
    Mede a fração de pixels 'de tinta' dentro da célula, com uma margem
    interna (margin_ratio) pra não contar a própria borda impressa do
    quadradinho como se fosse marcação do votante.
    """
    x1, y1, x2, y2 = cell
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin_ratio), int(h * margin_ratio)
    roi = binary_img[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / roi.size


# ---------------------------------------------------------------------------
# 4. Leitura de uma fileira (um filme) e da cédula inteira
# ---------------------------------------------------------------------------

def read_row(binary_img, row_cfg, fill_threshold):
    """
    Lê uma fileira de 10 quadradinhos e decide a nota marcada.

    Retorna (nota_ou_None, status, lista_de_fill_ratios)
    status: "ok" | "sem_marcacao" | "multiplas_marcacoes"
    """
    cells = build_row_cells(row_cfg)
    margin = row_cfg.get("margin_ratio", 0.18)
    ratios = [cell_fill_ratio(binary_img, c, margin) for c in cells]

    marked = [i for i, r in enumerate(ratios) if r >= fill_threshold]

    if len(marked) == 0:
        return None, "sem_marcacao", ratios
    if len(marked) > 1:
        return None, "multiplas_marcacoes", ratios

    nota = marked[0] + 1  # quadradinhos são 1..10, índice da lista é 0..9
    return nota, "ok", ratios


def read_ballot(image, config, canvas_size=None, manual_corners=None):
    """
    Pipeline completo para UMA imagem de cédula.

    Retorna um dicionário:
      {
        "status": "ok" | "papel_nao_encontrado" | "revisar",
        "notas": {"filme1": int|None, "filme2": int|None, ...},
        "detalhes": {"filme1": {"status":..., "ratios":[...]}, ...},
        "debug_image": imagem BGR com overlay pra conferência visual
      }
    """
    canvas_size = tuple(config.get("canvas_size", canvas_size or (1000, 700)))
    warped = warp_document(image, canvas_size=canvas_size, manual_corners=manual_corners)

    if warped is None:
        return {
            "status": "papel_nao_encontrado",
            "notas": {},
            "detalhes": {},
            "debug_image": image,
        }

    binary = binarize(warped)

    notas = {}
    detalhes = {}
    overall_status = "ok"

    for row_cfg in config["rows"]:
        label = row_cfg["label"]
        nota, status, ratios = read_row(binary, row_cfg, config.get("fill_threshold", 0.35))
        notas[label] = nota
        detalhes[label] = {"status": status, "ratios": ratios}
        if status != "ok":
            overall_status = "revisar"

    debug_image = draw_debug_overlay(warped, config, detalhes)

    return {
        "status": overall_status,
        "notas": notas,
        "detalhes": detalhes,
        "debug_image": debug_image,
    }


# ---------------------------------------------------------------------------
# 5. Overlay visual (pra você conferir/calibrar com os próprios olhos)
# ---------------------------------------------------------------------------

def draw_debug_overlay(warped_bgr, config, detalhes):
    """Desenha os 20 quadradinhos sobre a imagem corrigida:
       verde = marcado, cinza = vazio, vermelho = ambíguo/erro na fileira."""
    out = warped_bgr.copy()
    threshold = config.get("fill_threshold", 0.35)

    for row_cfg in config["rows"]:
        label = row_cfg["label"]
        cells = build_row_cells(row_cfg)
        ratios = detalhes[label]["ratios"]
        row_status = detalhes[label]["status"]

        for i, (cell, ratio) in enumerate(zip(cells, ratios)):
            x1, y1, x2, y2 = cell
            marked = ratio >= threshold
            if row_status == "ok" and marked:
                color = (0, 200, 0)          # verde
                thickness = 3
            elif row_status != "ok" and marked:
                color = (0, 0, 255)          # vermelho (ambíguo)
                thickness = 3
            else:
                color = (140, 140, 140)      # cinza
                thickness = 1
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(out, str(i + 1), (x1 + 3, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        cv2.putText(out, label, (row_cfg["x_start"], row_cfg["y_start"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2, cv2.LINE_AA)

    return out
