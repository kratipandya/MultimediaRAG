# use PowerShell instead of sh:
set shell := ["powershell.exe", "-c"]


# Resets and builds the embeddings except arxiv paper
build-faiss:
    cd backend/RAGembedder | py -3.12 .\build.py

# Zero-shot to test that embeddings work
zero-shot:
    cd backend/RAGembedder | py -3.12 .\zero_shot.py

# Run backend with dev mode
run-back:
    cd backend | py -3.12 .\api.py

dev-back:
    cd backend | py -3.12 -m robyn .\api.py --dev



run-front:
    cd frontend | pnpm run

# Run frontend with dev mode
dev-front:
    cd frontend | pnpm run dev

clear:
    rd /s /q node_modules | del package-lock.json

install:
    cd frontend/react-web | pnpm install

clear-install:
  just clear
  just install

# Get line count of all subdirectories, including .git-folder.
linecount type:
    dir . -filter "*{{type}}" -Recurse -name | foreach{(GC $_).Count} | measure-object -sum
