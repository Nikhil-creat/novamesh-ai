# Publish NovaMesh AI on GitHub Pages

Live URL after setup: `https://nikhil-creat.github.io/novamesh-ai/`

## 1. Create the repository (browser)
1. github.com → **New repository** → name `novamesh-ai` → Public → **Create** (leave it empty).

## 2. Upload from phone with Termux
```bash
pkg install git unzip -y
cd ~/storage/downloads            # run termux-setup-storage once if needed
unzip novamesh-ai.zip && cd novamesh-ai
git init && git add . && git commit -m "NovaMesh AI v2"
git branch -M main
git remote add origin https://github.com/Nikhil-creat/novamesh-ai.git
git push -u origin main           # username + a Personal Access Token (repo scope) as password
```
Token: GitHub → Settings → Developer settings → Personal access tokens → generate with `repo` scope.

## 3. Upload without Git (browser only)
Repo → **Add file → Upload files** → drag the contents of the unzipped folder (keep `docs/` and `backend/`) → Commit.
`.nojekyll` is hidden; if it is missing, add an empty file with that name inside `docs/`.

## 4. Turn on Pages
Repo → **Settings → Pages** → Source: **Deploy from a branch** → Branch `main`, folder **/docs** → Save.
Wait 1–2 minutes, then open the live URL. Updates: push again and Pages rebuilds automatically.

## 5. Using the AI Copilot on the site
Paste a free Groq key from console.groq.com in the Copilot box. It is kept in sessionStorage only and sent to api.groq.com. Never commit keys to the repo.

## 6. Run the real backend (optional, needs Docker)
```bash
cd backend && cp .env.example .env     # fill keys
docker compose --env-file .env -f docker/docker-compose.yml up --build -d
```
GitHub Pages hosts only the static site; the backend needs a machine or cloud VM with Docker.

## Troubleshooting
- 404: check Pages source is `/docs` and `index.html` is directly inside it.
- Push rejected: use the token, not your password; run `git pull --rebase origin main` if the repo was not empty.
- Copilot error 401: the Groq key is wrong or expired.
