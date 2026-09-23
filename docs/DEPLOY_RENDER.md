# Deploying the PJME Forecast API on Render

This guide takes the GitHub repo [`pankaj-cod/Hybrid_ML`](https://github.com/pankaj-cod/Hybrid_ML)
and deploys it as a live web API on [Render](https://render.com) (free tier).
Total time is about 10 minutes, and most of that is waiting for the first build.

**What gets deployed:** the Docker image defined in `Dockerfile`. It serves the trained
hybrid model (`artifacts/model.joblib`) through FastAPI. The server loads the saved model
and does **not** retrain.

| Endpoint | What it does |
|---|---|
| `GET /` | Service info |
| `GET /health` | Model status, training range and test metrics |
| `GET /docs` | Interactive API page (try requests in the browser) |
| `GET /forecast/latest?horizon=24` | Forecast after the end of the bundled dataset (demo) |
| `POST /forecast` | Forecast from history you send (≥ 168 hourly values) |

---

## 0. Before you start (checklist)

- [ ] The code is on GitHub on branch `main`. It is: commit `eb8d591`.
- [ ] CI is green. Check the **Actions** tab on GitHub; both the `test` and `docker` jobs
      should show ✅. The `docker` job builds and starts this exact container, so a
      green CI means the image works on Linux.
- [ ] These files are in the repo root: `Dockerfile`, `render.yaml`, `requirements.txt`,
      `artifacts/model.joblib`.

You do **not** need Docker installed on your computer. Render builds the image itself.

---

## 1. Create a Render account

1. Go to <https://render.com> and click **Get Started** (or **Sign In**).
2. Choose **GitHub** as the sign-in method. This links Render to your GitHub account
   in one step.
3. GitHub asks you to authorize Render. Click **Authorize Render**.
4. Finish any onboarding questions. You land on the Render **Dashboard**.

> The free tier does not need a credit card.

---

## 2. Give Render access to the repository

Render can only see repos you allow it to.

1. In the Dashboard, click **+ New** (top right) → **Blueprint**.
2. If Hybrid_ML is not listed, click **Configure account** (or
   **Connect GitHub → Configure**). This opens GitHub's app-permission page.
3. Under **Repository access**, choose either:
   - **Only select repositories** → pick `pankaj-cod/Hybrid_ML` (recommended), or
   - **All repositories**.
4. Click **Save**. GitHub sends you back to Render, and `Hybrid_ML` now appears in the list.

---

## 3. Deploy with the Blueprint (recommended)

The repo contains `render.yaml`, which tells Render exactly what to create:

```yaml
services:
  - type: web
    name: pjme-forecast
    runtime: docker        # build from ./Dockerfile
    plan: free
    healthCheckPath: /health
    autoDeploy: true       # redeploy on every push to main
```

Steps:

1. **+ New → Blueprint**, then click **Connect** next to `pankaj-cod/Hybrid_ML`.
2. **Blueprint Name:** type anything, e.g. `hybrid-ml`.
3. **Branch:** `main`.
4. Render shows a preview: *1 new Web Service: `pjme-forecast` (Docker, Free)*.
5. Click **Apply** (or **Deploy Blueprint**).

You don't need to set any environment variables. `PORT`, `PJME_MODEL_PATH` and
`PJME_HISTORY_PATH` all have defaults in the Dockerfile, and Render sets `PORT` itself.

> If the service name `pjme-forecast` is taken, Render adds a random suffix to the URL.
> That's fine.

---

## 4. Watch the build

1. Click the new **pjme-forecast** service, then open the **Logs** (or **Events**) tab.
2. The first build takes about **3–6 minutes**. You'll see roughly:
   ```
   ==> Building with Dockerfile
   #8 RUN pip install --no-cache-dir -r requirements.txt
   ...
   ==> Deploying...
   INFO:     Loaded hybrid model trained on ['2002-01-01 01:00:00', '2018-08-03 00:00:00']
   INFO:     Uvicorn running on http://0.0.0.0:10000
   ==> Your service is live 🎉
   ```
3. The status badge at the top turns green: **Live**.
4. Your URL appears under the service name, e.g.
   `https://pjme-forecast.onrender.com` (or with a suffix like `-a1b2`).

If the build fails, go to **Troubleshooting** below.

---

## 5. Test the live service

Replace `URL` with your service URL.

### In the browser
- `URL/health` shows JSON with `"status": "ok"` and the model's test metrics.
- `URL/docs` opens an interactive page. Click an endpoint → **Try it out** → **Execute**.
- `URL/forecast/latest?horizon=24` returns 24 hourly forecasts.

> The bundled dataset ends on **2018-08-03**, so `/forecast/latest` forecasts
> 2018-08-03 01:00 → 2018-08-04 00:00. It's a demo. For real use, send recent data to
> `POST /forecast`.

### From the terminal

```bash
curl https://YOUR-URL.onrender.com/health
```

```bash
curl "https://YOUR-URL.onrender.com/forecast/latest?horizon=24"
```

### POST your own history (Python)

`POST /forecast` needs **at least 168 consecutive hourly values** (7 days), and they must
end right before the hours you want to forecast.

```python
import pandas as pd, requests

URL = "https://YOUR-URL.onrender.com"
df = pd.read_csv("data/raw/PJME_hourly.csv", parse_dates=["Datetime"]).sort_values("Datetime")
last_week = df.tail(200)  # any >= 168 recent hours

body = {
    "history": [{"timestamp": t.isoformat(), "value": float(v)}
                for t, v in zip(last_week.Datetime, last_week.PJME_MW)],
    "horizon": 24,
}
r = requests.post(f"{URL}/forecast", json=body, timeout=120)
r.raise_for_status()
print(pd.DataFrame(r.json()["forecast"]))
```

Errors you might get back:
- `422 "Need at least 168 hours of history"`: send more data.
- `422 "Found a gap of N consecutive missing hours"`: your history has a hole longer
  than 6 hours. Fill it or send a later window.

---

## 6. Updating the deployment

`autoDeploy: true` means **every push to `main` triggers a new deploy automatically.**

### Code change
```bash
git add -A && git commit -m "describe the change" && git push
```
Render rebuilds and switches over when the new version passes `/health`. The old
version keeps serving until then, so there's no downtime.

### Retrain the model (new data or config)
```bash
.venv/bin/pjme train                      # rewrites artifacts/model.joblib + reports
.venv/bin/python -m pytest -q             # make sure tests still pass
git add artifacts config && git commit -m "Retrain model" && git push
```

> **Important:** the model file is a pickle. It must be loaded with the same
> `scikit-learn` and `lightgbm` versions that trained it, and those are pinned in
> `requirements.txt`. If you upgrade either library, retrain **and** update the pins in
> the same commit.

### Manual redeploy / rollback
- Service page → **Manual Deploy → Deploy latest commit**.
- Service page → **Events**, pick an earlier successful deploy → **Rollback**.

---

## 7. Free-tier behaviour (what to expect)

| Thing | Free tier |
|---|---|
| Sleep | The service **spins down after ~15 min without traffic**. The next request wakes it and takes about **30–60 s**. Later requests are fast. |
| Resources | 512 MB RAM, shared CPU. That's enough for this API. |
| Hours | Free instance hours are capped per month. One always-on service fits. |
| Custom domain | Supported: **Settings → Custom Domains**. |

To avoid cold starts, upgrade the instance type in **Settings → Instance Type** (paid)
or use an external uptime pinger on `/health`.

---

## 8. Troubleshooting

| Symptom (in Render logs) | Cause | Fix |
|---|---|---|
| `Hybrid_ML` not shown when creating the Blueprint | Render has no access to the repo | Step 2: **Configure account** on GitHub and add the repo |
| `failed to read dockerfile` / `render.yaml not found` | Files not on `main`, or wrong branch picked | Check the files are in the repo root on `main`; re-select branch `main` |
| `ERROR: No matching distribution found for …` during `pip install` | A pinned version isn't available for the image's Python | Keep `FROM python:3.12-slim` in the Dockerfile (CI tests this exact setup) |
| `OSError: libgomp.so.1: cannot open shared object file` | OpenMP missing | The Dockerfile installs `libgomp1`; make sure that line wasn't removed |
| `FileNotFoundError: artifacts/model.joblib` | Model not committed | `git add artifacts/model.joblib && git commit && git push` (check `.gitignore` doesn't exclude `artifacts/`) |
| `InconsistentVersionWarning` or `AttributeError` when unpickling | Library versions differ from training | Retrain with the pinned versions, or restore the pins in `requirements.txt` |
| Deploy stuck on "health check failed" | App crashed on startup | Read the first Python traceback in **Logs**; it's usually one of the rows above |
| `Out of memory (used over 512Mi)` | Free instance too small | Upgrade the instance type, or remove `PJME_HISTORY_PATH` (disables `/forecast/latest`) to save memory |
| First request takes ~1 minute | Free tier waking from sleep | Expected. See section 7 |

Still stuck? Copy the **first error block** from the Logs tab. The first traceback is
the real cause; later ones are usually side effects.

---

## Alternative: create the service manually (without the Blueprint)

If you'd rather not use `render.yaml`:

1. **+ New → Web Service** → connect `pankaj-cod/Hybrid_ML`.
2. **Language / Runtime:** `Docker`.
3. **Branch:** `main`. **Root directory:** leave empty.
4. **Instance type:** `Free`.
5. **Advanced → Health Check Path:** `/health`.
6. Click **Create Web Service**.

The result is identical. Render uses the same Dockerfile.
