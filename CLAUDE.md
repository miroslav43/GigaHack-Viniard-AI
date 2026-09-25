# GigaHack 2026 — Vineyard AI Field Challenge (Sireț3) · echipa Solemtrix

Repo-ul echipei: model AI + pipeline (`src/AI/`), aplicația web (`src/Web/`), date de la organizatori (`data & info/`, fișierele mari sunt în `.gitignore`).

## Web

**Web:** tot codul, configul, asset-urile, testele, documentația și dependențele aplicației web trăiesc exclusiv în `src/Web/`. Nu crea și nu modifica fișiere în afara `src/Web/` pentru partea web. Datele produse de pipeline-ul AI se citesc din locațiile din `src/Web/CLAUDE.md` → „Data contracts”, nu se copiază manual în alte locuri.

Singurele excepții în afara `src/Web/`:
1. această secțiune;
2. workflow-ul de CI `.github/workflows/web.yml`, care rulează doar verificările din `src/Web/`;
3. la implementare, linkul și instrucțiunile de rulare ale interfeței în `README.md`-ul din rădăcină (cerut de regulament).

Detalii: `src/Web/CLAUDE.md` și `src/Web/docs/PLAN.md`.
