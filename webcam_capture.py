"""
webcam_capture.py
------------------
Modo "ao vivo": abre a webcam, tenta ler a cédula em cada frame e mostra
overlay em tempo real (quadradinho verde = marcado, vermelho = ambíguo).
Quando a leitura estiver estável (mesmo resultado por alguns frames
seguidos) e for válida, aperte ESPAÇO pra confirmar e salvar o voto no CSV.

Uso:
    python webcam_capture.py [indice_da_camera]

Controles:
    ESPAÇO -> confirma e salva o voto atual (só funciona se status = ok)
    q      -> sai
"""

import sys
import csv
from pathlib import Path
from collections import deque

import cv2

from omr_core import read_ballot, load_config

OUTPUT_CSV = Path("resultados_webcam.csv")


def append_result(labels, notas):
    novo = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=labels)
        if novo:
            writer.writeheader()
        writer.writerow(notas)


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config = load_config("grid_config.json")
    labels = [g["label"] for g in config["groups"]]

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir a câmera de índice {cam_index}.")

    print("Aponte a cédula pra câmera. ESPAÇO = confirmar voto | q = sair.")

    # guarda as últimas leituras pra só liberar confirmação quando estabilizar
    historico = deque(maxlen=8)
    contador_salvos = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao ler frame da câmera.")
            break

        result = read_ballot(frame, config)
        display = result["debug_image"]

        estavel = False
        if result["status"] == "ok":
            historico.append(tuple(result["notas"][l] for l in labels))
            estavel = len(historico) == historico.maxlen and len(set(historico)) == 1
        else:
            historico.clear()

        cor_status = (0, 200, 0) if estavel else (0, 165, 255)
        texto_status = "PRONTO - aperte ESPAÇO" if estavel else f"status: {result['status']}"
        cv2.putText(display, texto_status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, cor_status, 2, cv2.LINE_AA)
        cv2.putText(display, f"votos salvos nesta sessao: {contador_salvos}",
                    (15, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("leitor de cedulas - Bonito", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord(' ') and estavel:
            append_result(labels, result["notas"])
            contador_salvos += 1
            print(f"Voto salvo: {result['notas']}")
            historico.clear()
            cv2.waitKey(400)  # pequena pausa pra trocar de cédula sem duplicar leitura

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFim da sessão. {contador_salvos} votos salvos em {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
