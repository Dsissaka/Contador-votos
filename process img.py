"""
process_images.py
------------------
Processa em lote uma pasta de fotos/scans de cédulas já capturadas
(webcam, celular, scanner) e gera um CSV com os votos + imagens de
depuração pra conferência manual dos casos duvidosos.

Uso:
    python process_images.py pasta_com_fotos/ [pasta_de_saida/]

Pré-requisito: já ter rodado calibrate.py e ter um grid_config.json
na mesma pasta deste script.
"""

import sys
import csv
from pathlib import Path

import cv2

from omr_core import read_ballot, load_config

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    if len(sys.argv) < 2:
        sys.exit("Uso: python process_images.py pasta_com_fotos/ [pasta_de_saida/]")

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else input_dir / "resultados"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config("grid_config.json")

    images = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not images:
        sys.exit(f"Nenhuma imagem encontrada em {input_dir}")

    rows_out = []
    labels = [r["label"] for r in config["rows"]]

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[AVISO] não consegui abrir {path.name}, pulando.")
            continue

        result = read_ballot(image, config)

        row = {"arquivo": path.name, "status": result["status"]}
        for label in labels:
            row[label] = result["notas"].get(label)
        rows_out.append(row)

        # salva imagem de depuração só pra revisar (ajuda MUITO a calibrar
        # o fill_threshold e a conferir os casos "revisar")
        debug_path = output_dir / f"debug_{path.stem}.jpg"
        cv2.imwrite(str(debug_path), result["debug_image"])

        marca = "OK" if result["status"] == "ok" else "REVISAR"
        print(f"{path.name}: {row} [{marca}]")

    csv_path = output_dir / "resultados.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arquivo", "status"] + labels)
        writer.writeheader()
        writer.writerows(rows_out)

    n_ok = sum(1 for r in rows_out if r["status"] == "ok")
    n_revisar = len(rows_out) - n_ok
    print(f"\nConcluído: {len(rows_out)} cédulas processadas "
          f"({n_ok} ok, {n_revisar} para revisar manualmente).")
    print(f"CSV salvo em: {csv_path}")
    print(f"Imagens de depuração em: {output_dir}")


if __name__ == "__main__":
    main()
