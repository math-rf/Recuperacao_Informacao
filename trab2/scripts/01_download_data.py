"""
01_download_data.py
===================
Baixa e extrai o dataset NFCorpus (BEIR) para o diretório `data/nfcorpus`.

Passo 1 do pipeline. Executar uma única vez:

    python scripts/01_download_data.py

Arquivos gerados:
    data/nfcorpus/corpus.jsonl
    data/nfcorpus/queries.jsonl
    data/nfcorpus/qrels/dev.tsv
    data/nfcorpus/qrels/test.tsv
"""

import sys
import zipfile
from pathlib import Path

# Permite importar `config` a partir da raiz do projeto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def download_zip(url: str, dest: Path) -> None:
    """Baixa um arquivo via streaming (sem carregar tudo na memória)."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Baixando {url} ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"      {done / total * 100:5.1f}%", end="\r")
    print(f"\n      Concluído: {dest}")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extrai o zip preservando a estrutura interna (pasta `nfcorpus/`)."""
    print(f"[2/3] Extraindo {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print(f"      Extraído em: {dest_dir}")


def main() -> None:
    zip_path = config.DATA_DIR.parent / "nfcorpus.zip"

    # Já baixado?
    if config.CORPUS_FILE.exists() and config.QUERIES_FILE.exists():
        print("Dataset já presente. Nada a fazer.")
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        download_zip(config.NFCORPUS_URL, zip_path)
    else:
        print("[1/3] Zip já baixado.")

    extract_zip(zip_path, config.DATA_DIR.parent)

    # Verificação final.
    print("[3/3] Verificando arquivos:")
    for f in [config.CORPUS_FILE, config.QUERIES_FILE,
              config.QRELS_DEV_FILE, config.QRELS_TEST_FILE]:
        status = "OK " if f.exists() else "FALTA"
        print(f"      [{status}] {f.name}")


if __name__ == "__main__":
    main()
