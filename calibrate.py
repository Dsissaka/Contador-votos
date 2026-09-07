"""
calibrate.py
------------
Ferramenta de calibração de UMA vez só. Você roda isso com uma foto de
exemplo de uma cédula (de preferência tirada nas mesmas condições que vão
ser usadas no festival: mesma mesa, mesma câmera/webcam), clica em alguns
pontos com o mouse, e o script gera o arquivo `grid_config.json` que o resto
do sistema (process_images.py e webcam_capture.py) vai usar.

Uso:
    python calibrate.py caminho/da/foto_exemplo.jpg

Passos na tela:
    1. Se o papel não for detectado automaticamente, clique nos 4 cantos
       do papel na ordem: superior-esquerdo, superior-direito,
       inferior-direito, inferior-esquerdo. Aperte 'n' se a detecção
       automática já estiver correta (contorno verde) e quiser aceitá-la.
    2. Na imagem já 'endireitada', clique no canto superior-esquerdo e depois
       no canto inferior-direito da FILEIRA DE QUADRADINHOS DO FILME 1
       (ou seja: 2 cliques cobrindo do quadradinho 1 até o quadradinho 10).
    3. Repita para o FILME 2.
    4. Confira o preview com o grid desenhado. Aperte 'y' para salvar,
       'r' para clicar tudo de novo, ou 'q' para sair sem salvar.
"""

import sys
import json
import cv2

from omr_core import find_document_contour, four_point_transform, order_points

CANVAS_SIZE = (1000, 700)
WINDOW = "calibracao"


def click_points(image, n_points, instructions):
    """Mostra a imagem, coleta n_points cliques do mouse, devolve a lista."""
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
    """Detecta o papel automaticamente; se falhar (ou usuário recusar),
    pede 4 cliques manuais nos cantos."""
    auto = find_document_contour(image)
    if auto is not None:
        preview = image.copy()
        cv2.polylines(preview, [auto.astype(int)], True, (0, 255, 0), 3)
        cv2.imshow(WINDOW, preview)
        print("Papel detectado automaticamente (contorno verde).")
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


def calibrate_row(warped, label):
    pts = click_points(
        warped, 2,
        f"[{label}] Clique no canto SUPERIOR-ESQUERDO do quadradinho 1, "
        f"depois no canto INFERIOR-DIREITO do quadradinho 10."
    )
    (x1, y1), (x2, y2) = pts
    return {
        "label": label,
        "x_start": min(x1, x2),
        "x_end": max(x1, x2),
        "y_start": min(y1, y2),
        "y_end": max(y1, y2),
        "n_boxes": 10,
        "margin_ratio": 0.18,
    }


def preview_grid(warped, rows, fill_threshold):
    # import local pra evitar dependência circular na leitura do módulo
    from omr_core import build_row_cells
    out = warped.copy()
    for row in rows:
        for i, (x1, y1, x2, y2) in enumerate(build_row_cells(row)):
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 128, 0), 1)
        cv2.putText(out, row["label"], (row["x_start"], row["y_start"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
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

        row1 = calibrate_row(warped, "filme1")
        row2 = calibrate_row(warped, "filme2")

        fill_threshold = 0.35
        preview = preview_grid(warped, [row1, row2], fill_threshold)
        cv2.putText(preview, "y = salvar | r = refazer | q = sair sem salvar",
                    (10, CANVAS_SIZE[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)
        cv2.imshow(WINDOW, preview)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('y'):
            config = {
                "canvas_size": list(CANVAS_SIZE),
                "fill_threshold": fill_threshold,
                "rows": [row1, row2],
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
