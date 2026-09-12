"""
omr_core.py
------------
Núcleo do algoritmo de leitura das cédulas do festival de Bonito.

Pipeline:
  1. Localizar o papel na foto e corrigir a perspectiva.
     Duas estratégias, nessa ordem de preferência:
       a) Marcadores de canto: 4 quadradinhos pretos desenhados perto dos
          4 cantos do papel. Detectados como blobs independentes -> não
          exige que a borda do papel esteja inteira/visível (funciona
          mesmo com a mão ou outros objetos tampando parte da borda).
       b) Contorno do papel inteiro (método antigo, via detecção de
          bordas) -> usado só como fallback se não achar os 4 marcadores.
  2. Binarizar (separar tinta de fundo, tolerando iluminação desigual).
  3. Usar as coordenadas de grid_config.json (geradas por calibrate.py)
     pra recortar cada quadradinho de cada grupo (filme). Cada grupo pode
     ser uma fileira horizontal ou uma coluna vertical de caixas.
  4. Decidir qual quadradinho foi marcado de forma RELATIVA (o quadradinho
     com claramente mais tinta que os outros da mesma fileira/coluna
     vence) em vez de um limiar fixo -> funciona tanto pra preenchimento
     completo quanto pra um X ou um risco "/".
"""

import json
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Utilitário geométrico
# ---------------------------------------------------------------------------

def order_points(pts):
    """Ordena 4 pontos como [topo-esq, topo-dir, baixo-dir, baixo-esq]."""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def four_point_transform(image, pts, canvas_size):
    rect = order_points(pts)
    W, H = canvas_size
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (W, H))


def _flat_field_binary(gray, blur_sigma=25, extra_close=0):
    """Normaliza iluminação (flat-fielding, divide pela versão desfocada
    de si mesma) e binariza com Otsu. Usado na binarização da cédula já
    corrigida, na hora de medir tinta dentro de cada quadradinho."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if extra_close:
        kernel = np.ones((extra_close, extra_close), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


# ---------------------------------------------------------------------------
# 1a. Localizar o papel pela região clara (mais robusto que bordas/Canny)
# ---------------------------------------------------------------------------

def find_paper_mask(image, min_area_ratio=0.05, close_kernel=15):
    """
    Acha o papel como o maior "blob" claro (branco) da imagem, em vez de
    tentar seguir a borda do papel. Isso funciona muito melhor com mesa de
    madeira no fundo: mesa com veios escuros tem MUITA variação de tom
    parecida com tinta/sombra, o que confunde tanto detecção de bordas
    (Canny) quanto blobs escuros — mas o papel continua sendo, disparado,
    a maior região CLARA e uniforme da cena.

    Retorna (mask, contour) do maior componente claro, ou (None, None).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # fecha buracos pequenos (marcadores de canto, texto, linhas do grid
    # ficam "escuros" dentro do papel e virariam buracos na máscara)
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n_labels <= 1:
        return None, None

    img_area = gray.shape[0] * gray.shape[1]
    best_label, best_area = None, 0
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area, best_label = area, label

    if best_area / img_area < min_area_ratio:
        return None, None

    mask = np.uint8(labels == best_label) * 255
    # preenche buracos internos remanescentes (números, marcadores) antes
    # de extrair o contorno, pra pegar só o contorno EXTERNO do papel
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    biggest_contour = max(contours, key=cv2.contourArea)
    return mask, biggest_contour


# ---------------------------------------------------------------------------
# 1b. Detecção via marcadores de canto (método preferido quando presentes)
# ---------------------------------------------------------------------------

def find_corner_markers(image, area_ratio_range=(0.0004, 0.08),
                         aspect_range=(0.35, 2.8), min_extent=0.25):
    """
    Procura 4 blobs pretos sólidos, um em cada canto do papel (os
    quadradinhos rabiscados nos cantos da cédula).

    A busca é restrita à região do papel (achada via find_paper_mask)
    quando possível — isso evita confundir veios escuros da mesa (fora do
    papel) com marcadores, que era o problema em fotos com fundo de
    madeira.

    Retorna 4 pontos [topo-esq, topo-dir, baixo-dir, baixo-esq] (centro de
    cada marcador) ou None se não conseguir identificar os 4 com confiança.
    """
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_full.shape

    # restringe a busca à região do papel (com uma margem), se achada
    ox, oy = 0, 0
    mask, contour = find_paper_mask(image)
    if contour is not None:
        x, y, bw, bh = cv2.boundingRect(contour)
        pad = int(0.06 * max(bw, bh))
        ox, oy = max(0, x - pad), max(0, y - pad)
        ex, ey = min(w, x + bw + pad), min(h, y + bh + pad)
        gray = gray_full[oy:ey, ox:ex]
    else:
        gray = gray_full

    img_area = gray.shape[0] * gray.shape[1]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # connectedComponents mede área de pixels de verdade por região. Usar
    # contornos aqui seria enganoso: se o fundo escuro em volta do papel
    # tocar as 4 bordas da foto, o CONTORNO externo desse fundo abrange a
    # imagem inteira (é uma "moldura"), mesmo que a área de pixels escuros
    # seja pequena — o que faria o filtro de tamanho falhar silenciosamente.
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    for label in range(1, n_labels):  # pula o rótulo 0 (fundo)
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= 0:
            continue
        ratio = area / img_area
        if not (area_ratio_range[0] <= ratio <= area_ratio_range[1]):
            continue
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        if bw == 0 or bh == 0:
            continue
        aspect = bw / bh
        if not (aspect_range[0] <= aspect <= aspect_range[1]):
            continue
        extent = area / float(bw * bh)
        if extent < min_extent:
            continue
        cx, cy = centroids[label]
        candidates.append({"center": (float(cx) + ox, float(cy) + oy), "area": float(area)})

    if len(candidates) < 4:
        return None

    # descarta candidatos com área muito diferente da mediana (filtra
    # objetos aleatórios no fundo, tipo caneta ou embalagem)
    areas = sorted(c["area"] for c in candidates)
    median_area = areas[len(areas) // 2]
    candidates = [c for c in candidates if 0.25 <= c["area"] / median_area <= 4.0]
    if len(candidates) < 4:
        return None

    # 1 candidato por quadrante (relativo ao centroide de todos eles),
    # pegando o mais extremo (mais distante do centro) em cada quadrante
    cx_all = float(np.mean([c["center"][0] for c in candidates]))
    cy_all = float(np.mean([c["center"][1] for c in candidates]))

    quadrants = {"tl": None, "tr": None, "bl": None, "br": None}
    for c in candidates:
        x, y = c["center"]
        key = ("t" if y < cy_all else "b") + ("l" if x < cx_all else "r")
        dist = (x - cx_all) ** 2 + (y - cy_all) ** 2
        if quadrants[key] is None or dist > quadrants[key]["dist"]:
            quadrants[key] = {"center": c["center"], "dist": dist, "area": c["area"]}

    if any(v is None for v in quadrants.values()):
        return None

    # Confere se os 4 marcadores ESCOLHIDOS (não só o conjunto geral de
    # candidatos) têm área parecida entre si. Isso é o que evita cair em
    # falso-positivo: 4 pedacinhos de texto (números impressos, letras)
    # espalhados pela imagem podem, por acaso, cair um em cada quadrante,
    # mas normalmente têm tamanhos bem diferentes entre si — marcadores de
    # verdade, feitos do mesmo jeito, têm área bem mais consistente.
    quad_areas = [quadrants[k]["area"] for k in ("tl", "tr", "bl", "br")]
    if max(quad_areas) / min(quad_areas) > 3.0:
        return None

    return np.array([
        quadrants["tl"]["center"],
        quadrants["tr"]["center"],
        quadrants["br"]["center"],
        quadrants["bl"]["center"],
    ], dtype="float32")


# ---------------------------------------------------------------------------
# 1c. Detecção via contorno do papel inteiro (fallback)
# ---------------------------------------------------------------------------

def find_document_contour(image, min_area_ratio=0.2):
    """
    Encontra os 4 cantos do papel a partir da máscara de região clara
    (find_paper_mask). Usa o retângulo de área mínima (minAreaRect) em vez
    de tentar forçar o contorno a virar exatamente 4 pontos com
    approxPolyDP — isso tolera cantos meio dobrados/rasgados no papel
    (comuns em folha de caderno arrancada) sem perder a detecção.
    """
    _, contour = find_paper_mask(image, min_area_ratio=min_area_ratio)
    if contour is None:
        return None
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype("float32")


# ---------------------------------------------------------------------------
# 1. Junta os métodos
# ---------------------------------------------------------------------------

def warp_document(image, canvas_size=(1000, 700), manual_corners=None):
    """
    Retorna (imagem_corrigida, metodo) ou (None, None) se nada funcionar.
    metodo: "marcadores" | "contorno" | "manual"
    """
    if manual_corners is not None:
        return four_point_transform(image, manual_corners, canvas_size), "manual"

    corners = find_corner_markers(image)
    if corners is not None:
        return four_point_transform(image, corners, canvas_size), "marcadores"

    corners = find_document_contour(image)
    if corners is not None:
        return four_point_transform(image, corners, canvas_size), "contorno"

    return None, None


# ---------------------------------------------------------------------------
# 2. Binarização da cédula já corrigida (pra ler as marcações)
# ---------------------------------------------------------------------------

def binarize(warped_bgr):
    """
    threshold adaptativo local quebra com marcações grandes e uniformes
    (o "fundo local" dentro da própria mancha de tinta já é escuro, então
    ela some). Em vez disso: normaliza iluminação (flat-fielding) e usa
    threshold global (Otsu) -> funciona bem tanto pra sombra/luz desigual
    quanto pra qualquer estilo de marcação (preenchido, X, risco).
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    return _flat_field_binary(gray, blur_sigma=25, extra_close=0)


# ---------------------------------------------------------------------------
# 3. Grid calibrado -> células dos quadradinhos (horizontal OU vertical)
# ---------------------------------------------------------------------------

def load_config(path="grid_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_group_cells(group_cfg):
    """
    Calcula as coordenadas (x1,y1,x2,y2) dos n_boxes quadradinhos de um
    grupo (um filme). orientation define se os quadradinhos estão lado a
    lado (fileira horizontal) ou empilhados (coluna vertical).
    """
    orientation = group_cfg.get("orientation", "horizontal")
    n = group_cfg["n_boxes"]
    x_start, x_end = group_cfg["x_start"], group_cfg["x_end"]
    y_start, y_end = group_cfg["y_start"], group_cfg["y_end"]

    cells = []
    if orientation == "horizontal":
        cell_w = (x_end - x_start) / n
        for i in range(n):
            x1 = int(round(x_start + i * cell_w))
            x2 = int(round(x_start + (i + 1) * cell_w))
            cells.append((x1, y_start, x2, y_end))
    else:  # vertical
        cell_h = (y_end - y_start) / n
        for i in range(n):
            y1 = int(round(y_start + i * cell_h))
            y2 = int(round(y_start + (i + 1) * cell_h))
            cells.append((x_start, y1, x_end, y2))
    return cells


def build_group_cells_refined(binary_img, group_cfg):
    """Como build_group_cells, mas ajustando as linhas divisórias pra tinta
    real (ver refine_dividers) em vez de espaçamento perfeitamente igual."""
    orientation = group_cfg.get("orientation", "horizontal")
    dividers = refine_dividers(binary_img, group_cfg)
    if dividers is None:
        return build_group_cells(group_cfg)

    x_start, x_end = group_cfg["x_start"], group_cfg["x_end"]
    y_start, y_end = group_cfg["y_start"], group_cfg["y_end"]

    cells = []
    for i in range(len(dividers) - 1):
        a, b = dividers[i], dividers[i + 1]
        if orientation == "horizontal":
            cells.append((a, y_start, b, y_end))
        else:
            cells.append((x_start, a, x_end, b))
    return cells


def cell_fill_ratio(binary_img, cell, margin_along=0.12, margin_cross=0.12, orientation="horizontal"):
    """
    Fração de pixels de tinta dentro da célula, com margem interna — mas
    com DUAS margens diferentes, não uma só:

    - margin_along: margem na direção em que as células se dividem (topo/
      base de uma coluna vertical, ou esquerda/direita de uma fileira
      horizontal). Só precisa ser grande o suficiente pra excluir a linha
      divisória entre uma célula e a próxima.
    - margin_cross: margem na direção PERPENDICULAR — as bordas laterais
      de uma coluna (ou topo/base de uma fileira) são UMA linha contínua
      que atravessa TODAS as células, não uma linha por célula. Se essa
      margem for pequena demais, toda célula (marcada ou não) herda um
      "chão" de tinta vindo dessa borda, o que atrapalha bastante a
      comparação relativa entre células — por isso essa margem por
      padrão é maior que margin_along.
    """
    x1, y1, x2, y2 = cell
    w, h = x2 - x1, y2 - y1
    if orientation == "horizontal":
        mx, my = max(1, int(w * margin_along)), max(1, int(h * margin_cross))
    else:
        mx, my = max(1, int(w * margin_cross)), max(1, int(h * margin_along))
    roi = binary_img[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / roi.size


def refine_dividers(binary_img, group_cfg, search_frac=0.35):
    """
    Quadradinhos desenhados à mão raramente têm exatamente o mesmo
    tamanho — numa cédula real, a primeira caixa de uma coluna pode ser
    visivelmente maior que as últimas. Se a gente simplesmente dividir o
    espaço calibrado em N partes IGUAIS, esse desalinhamento acumula e as
    últimas caixas ficam lendo o quadradinho errado.

    Em vez disso: parte do palpite de espaçamento igual, mas ajusta cada
    linha divisória procurando, numa janela pequena ao redor do palpite, a
    posição onde realmente tem uma linha de tinta desenhada (soma de
    pixels de tinta ao longo da largura/altura do grupo). Se não achar
    nada confiável na janela (ex: imagem sem contraste ali), mantém o
    palpite de espaçamento igual — então isso nunca piora o caso em que a
    divisão igual já funcionava.
    """
    orientation = group_cfg.get("orientation", "horizontal")
    n = group_cfg["n_boxes"]
    x_start, x_end = group_cfg["x_start"], group_cfg["x_end"]
    y_start, y_end = group_cfg["y_start"], group_cfg["y_end"]

    if orientation == "horizontal":
        axis_start, axis_end = x_start, x_end
        cross_lo, cross_hi = y_start, y_end
    else:
        axis_start, axis_end = y_start, y_end
        cross_lo, cross_hi = x_start, x_end

    axis_len = axis_end - axis_start
    if axis_len <= 0:
        return None

    h_img, w_img = binary_img.shape[:2]
    cross_lo, cross_hi = max(0, cross_lo), min(h_img if orientation == "horizontal" else w_img, cross_hi)
    axis_lo_clamped, axis_hi_clamped = max(0, axis_start), min(w_img if orientation == "horizontal" else h_img, axis_end)

    if orientation == "horizontal":
        strip = binary_img[cross_lo:cross_hi, axis_lo_clamped:axis_hi_clamped]
        profile = strip.sum(axis=0).astype(np.float64) if strip.size else np.zeros(axis_len)
    else:
        strip = binary_img[axis_lo_clamped:axis_hi_clamped, cross_lo:cross_hi]
        profile = strip.sum(axis=1).astype(np.float64) if strip.size else np.zeros(axis_len)

    if profile.shape[0] != axis_len:
        # a regiao calibrada saiu (parcialmente) da imagem; sem dado confiavel
        pad = axis_len - profile.shape[0]
        if pad > 0:
            profile = np.pad(profile, (0, pad))
        else:
            profile = profile[:axis_len]

    expected_step = axis_len / n
    search_radius = max(3, int(expected_step * search_frac))

    # busca SEQUENCIAL: o palpite de cada linha parte da ÚLTIMA linha já
    # encontrada (+ um passo esperado), não da posição "ideal" original.
    # Isso evita que o erro de uma caixa maior/menor que a média acumule
    # nas seguintes — cada divisória só precisa corrigir o desvio da
    # PRÓXIMA caixa, não o desvio de todas as anteriores somado.
    dividers = [0]
    for i in range(1, n):
        guess = dividers[-1] + expected_step
        guess = int(round(guess))
        lo = max(0, guess - search_radius)
        hi = min(axis_len, guess + search_radius + 1)
        window = profile[lo:hi]
        if window.size == 0 or window.max() <= 0:
            pos = guess
        else:
            pos = lo + int(np.argmax(window))
        dividers.append(max(pos, dividers[-1] + 1))
    dividers.append(max(axis_len, dividers[-1] + 1))

    return [axis_start + d for d in dividers]


# ---------------------------------------------------------------------------
# 4. Leitura de um grupo (um filme) — decisão RELATIVA, não por limiar fixo
# ---------------------------------------------------------------------------

def read_group(binary_img, group_cfg, min_ink_ratio=0.02, margin_factor=1.6):
    """
    Lê os N quadradinhos de um grupo e decide a nota marcada comparando o
    quadradinho com mais tinta contra o segundo colocado — em vez de exigir
    que ele passe de um limiar absoluto. Isso é o que permite aceitar tanto
    preenchimento completo quanto um X ou um risco fino: o que importa é
    ter *bem mais* tinta que os vizinhos, não uma quantidade absoluta.

    Retorna (nota_ou_None, status, lista_de_ratios)
    status: "ok" | "sem_marcacao" | "ambiguo"
    """
    cells = build_group_cells_refined(binary_img, group_cfg)
    orientation = group_cfg.get("orientation", "horizontal")
    margin_along = group_cfg.get("margin_ratio", 0.12)
    margin_cross = group_cfg.get("margin_ratio_cross", max(margin_along, 0.22))
    ratios = [cell_fill_ratio(binary_img, c, margin_along, margin_cross, orientation) for c in cells]

    order = sorted(range(len(ratios)), key=lambda i: ratios[i], reverse=True)
    best_idx = order[0]
    best_val = ratios[best_idx]
    second_val = ratios[order[1]] if len(order) > 1 else 0.0

    if best_val < min_ink_ratio:
        return None, "sem_marcacao", ratios

    if second_val >= min_ink_ratio and best_val < second_val * margin_factor:
        return None, "ambiguo", ratios

    return best_idx + 1, "ok", ratios


# ---------------------------------------------------------------------------
# 5. Pipeline completo de UMA cédula
# ---------------------------------------------------------------------------

def read_ballot(image, config, canvas_size=None, manual_corners=None):
    canvas_size = tuple(config.get("canvas_size", canvas_size or (1000, 700)))
    warped, metodo = warp_document(image, canvas_size=canvas_size, manual_corners=manual_corners)

    if warped is None:
        return {
            "status": "papel_nao_encontrado",
            "metodo_deteccao": None,
            "notas": {},
            "detalhes": {},
            "debug_image": image,
        }

    binary = binarize(warped)

    notas, detalhes = {}, {}
    overall_status = "ok"
    min_ink_ratio = config.get("min_ink_ratio", 0.02)
    margin_factor = config.get("margin_factor", 1.6)

    for group_cfg in config["groups"]:
        label = group_cfg["label"]
        nota, status, ratios = read_group(binary, group_cfg, min_ink_ratio, margin_factor)
        notas[label] = nota
        detalhes[label] = {"status": status, "ratios": ratios}
        if status != "ok":
            overall_status = "revisar"

    debug_image = draw_debug_overlay(warped, binary, config, detalhes, metodo)

    return {
        "status": overall_status,
        "metodo_deteccao": metodo,
        "notas": notas,
        "detalhes": detalhes,
        "debug_image": debug_image,
    }


# ---------------------------------------------------------------------------
# 6. Overlay visual de depuração
# ---------------------------------------------------------------------------

def draw_debug_overlay(warped_bgr, binary_img, config, detalhes, metodo=None):
    out = warped_bgr.copy()

    for group_cfg in config["groups"]:
        label = group_cfg["label"]
        cells = build_group_cells_refined(binary_img, group_cfg)
        ratios = detalhes[label]["ratios"]
        status = detalhes[label]["status"]
        best_idx = int(np.argmax(ratios)) if ratios else -1

        for i, (cell, ratio) in enumerate(zip(cells, ratios)):
            x1, y1, x2, y2 = cell
            is_best = (i == best_idx)
            if status == "ok" and is_best:
                color, thickness = (0, 200, 0), 3          # verde = escolhido
            elif status != "ok" and is_best:
                color, thickness = (0, 0, 255), 3           # vermelho = ambíguo/topo duvidoso
            else:
                color, thickness = (140, 140, 140), 1
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(out, str(i + 1), (x1 + 3, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        x0 = group_cfg["x_start"]
        y0 = max(15, group_cfg["y_start"] - 8)
        cv2.putText(out, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 128, 0), 2, cv2.LINE_AA)

    if metodo:
        cv2.putText(out, f"deteccao: {metodo}", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out
