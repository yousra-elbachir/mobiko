# Deploying the Triplet Annotator on RunAI / Kubernetes

The app is a single Streamlit process that serves all annotators concurrently and
saves their work to `ANNOTATION_SAVE_DIR`. For the cluster that means: **one replica**,
a **PVC** for persistence, and **inputs on the PVC** (so updating the extraction
files needs no rebuild).

SDSC-specific values (registry, storage class, ingress host/auth, project
namespace) — confirm with **@Albert / @Andrei** on Slack.

## 1. Build & push the image

```bash
cd code/annotation_app
docker build -t <IMAGE> .          # e.g. registry.<sdsc>/mobiko/annotator:1.0
docker push <IMAGE>
```

## 2. Create the PVC and load the inputs

```bash
kubectl apply -n <NAMESPACE> -f deploy/k8s.yaml      # creates the PVC (and the rest)

# Copy inputs into the volume via the running pod once it's up (step 3),
# or with a throwaway pod. Target layout on the PVC (/data):
#   /data/extracted_triplets/relations.json  <- shared input (Extract relations)
#   /data/extracted_triplets/triplets.json   <- shared input (Extract joint triplets)
#   /data/saves/                             <- created automatically; annotations + snapshots/
POD=$(kubectl get pod -n <NAMESPACE> -l app=mobiko-annotator -o name | head -1)
kubectl cp extracted_triplets <NAMESPACE>/${POD##*/}:/data/extracted_triplets
```

(If a task's file is missing the start screen shows an error for that task —
handy to confirm the deploy works before loading real data.)

## 3. Apply & access

```bash
kubectl apply -n <NAMESPACE> -f deploy/k8s.yaml
kubectl rollout status deploy/mobiko-annotator -n <NAMESPACE>
```

Open `https://<HOST>/`. To smoke-test without an ingress:
`kubectl port-forward -n <NAMESPACE> svc/mobiko-annotator 8501:80` → http://localhost:8501

## Updating the extraction files later

Copy the new JSON onto the PVC and restart — no rebuild:

```bash
POD=$(kubectl get pod -n <NAMESPACE> -l app=mobiko-annotator -o name | head -1)
kubectl cp extracted_triplets <NAMESPACE>/${POD##*/}:/data/extracted_triplets
kubectl rollout restart deploy/mobiko-annotator -n <NAMESPACE>
```

## Notes

- **Persistence**: annotations and `snapshots/` live under `/data/saves` on the PVC,
  so they survive pod restarts. Back up the PVC (or `kubectl cp` it out) periodically.
- **Single replica is intentional** — Streamlit keeps per-session state in process and
  writes the save files; a second replica would split sessions and race on writes.
- **Auth**: the in-app annotator name is just a label. If the SDSC ingress doesn't
  already enforce SSO, enable nginx basic-auth (see the commented block in `k8s.yaml`).
- **RunAI CLI alternative** (instead of raw kubectl): you can submit the same image as
  a RunAI workload exposing port 8501 with the PVC mounted at `/data`; the manifests
  above are the portable, explicit form and are easiest to keep in version control.
