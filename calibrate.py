"""
calibrate.py
------------
Ferramenta de calibração de UMA vez só. Rode isso com uma foto de exemplo
de uma cédula (de preferência nas mesmas condições reais que vão ser
usadas no evento), clique em alguns pontos com o mouse, e o script gera o
`grid_config.json` que o resto do sistema usa.

Uso:
    python calibrate.py caminho/da/foto_exemplo.jpg

Passos na tela:
    1. O script tenta achar os 4 marcadores de canto automaticamente
       (quadradinhos pretos nos cantos do papel). Se conseguir, mostra em
       verde — aperte 'n' pra aceitar. Se não conseguir (ou você recusar),
       clica manualmente nos 4 cantos do papel, na ordem: superior-esq,
       superior-dir, inferior-dir, inferior-esq.
    2. Na imagem já "endireitada", clique no canto superior-esquerdo e
       depois no canto inferior-direito do quadradinho 1 até o
       quadradinho 10 do FILME 1 (funciona tanto se os quadradinhos estão
       em fileira quanto em coluna — a orientação é detectada sozinha
       pelo formato do retângulo que você desenhar com os 2 cliques).
    3. Repita para o FILME 2.
    4. Confira o preview com o grid desenhado. Aperte 'y' pra salvar, 'r'
       pra refazer os cliques, ou 'q' pra sair sem salvar.
"""

import sys
import json
import cv2

from omr_core import (
    find_corner_markers,
    find_document_contour,
    four_point_transform,
    order_points,
    build_group_cells,
)

CANVAS_SIZE = (700, 1000)  # ajuste se o formato do seu papel for bem diferente
WINDOW = "calibracao"


def click_points(image, n_points, instructions):
    points = []
    display = image.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow(WINDOW, display)

    print(instructions)
    cv2.imshow(WINDOW, display)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while len(points) < n_points:
        if cv2.waitKey(20) & 0xFF == 27:  # ESC cancela
            sys.exit("Cancelado pelo usuário.")
    return points


def calibrate_document_corners(image):
    """Tenta os marcadores de canto primeiro, depois o contorno do papel
    inteiro, e só pede cliques manuais se os dois falharem."""
    auto = find_corner_markers(image)
    metodo = "marcadores"
    if auto is None:
        auto = find_document_contour(image)
        metodo = "contorno"

    if auto is not None:
        preview = image.copy()
        pts_int = auto.astype(int)
        cv2.polylines(preview, [pts_int], True, (0, 255, 0), 3)
        for (x, y) in pts_int:
            cv2.circle(preview, (x, y), 6, (0, 255, 0), -1)
        cv2.imshow(WINDOW, preview)
        print(f"Papel detectado automaticamente (método: {metodo}, contorno verde).")
        print("Aperte 'n' para aceitar, ou qualquer outra tecla para marcar manualmente.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            return order_points(auto)

    pts = click_points(
        image, 4,
        "Clique nos 4 cantos do papel, na ordem: "
        "superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo."
    )
    return order_points(pts)


def calibrate_group(warped, label):
    pts = click_points(
        warped, 2,
        f"[{label}] Clique no canto SUPERIOR-ESQUERDO do quadradinho 1, "
        f"depois no canto INFERIOR-DIREITO do quadradinho 10."
    )
    (x1, y1), (x2, y2) = pts
    x_start, x_end = min(x1, x2), max(x1, x2)
    y_start, y_end = min(y1, y2), max(y1, y2)

    # a orientação é inferida automaticamente: se a área clicada é mais
    # alta que larga, os quadradinhos estão empilhados numa coluna;
    # se é mais larga que alta, estão lado a lado numa fileira.
    orientation = "vertical" if (y_end - y_start) > (x_end - x_start) else "horizontal"

    return {
        "label": label,
        "orientation": orientation,
        "x_start": x_start, "x_end": x_end,
        "y_start": y_start, "y_end": y_end,
        "n_boxes": 10,
        "margin_ratio": 0.12,
        "margin_ratio_cross": 0.24,
    }


def preview_grid(warped, groups):
    out = warped.copy()
    for group in groups:
        for (x1, y1, x2, y2) in build_group_cells(group):
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 128, 0), 1)
        cv2.putText(out, f"{group['label']} ({group['orientation']})",
                    (group["x_start"], max(15, group["y_start"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python calibrate.py caminho/da/foto_exemplo.jpg")

    image = cv2.imread(sys.argv[1])
    if image is None:
        sys.exit(f"Não consegui abrir a imagem: {sys.argv[1]}")

    cv2.namedWindow(WINDOW)

    while True:
        corners = calibrate_document_corners(image)
        warped = four_point_transform(image, corners, CANVAS_SIZE)

        group1 = calibrate_group(warped, "filme1")
        group2 = calibrate_group(warped, "filme2")

        preview = preview_grid(warped, [group1, group2])
        cv2.putText(preview, "y = salvar | r = refazer | q = sair sem salvar",
                    (10, CANVAS_SIZE[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)
        cv2.imshow(WINDOW, preview)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('y'):
            config = {
                "canvas_size": list(CANVAS_SIZE),
                "min_ink_ratio": 0.02,
                "margin_factor": 1.6,
                "groups": [group1, group2],
            }
            with open("grid_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print("Salvo em grid_config.json")
            break
        elif key == ord('q'):
            print("Saindo sem salvar.")
            break
        # qualquer outra tecla ('r' incluso) -> refaz o loop

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
