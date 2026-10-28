"""main.py - Serveur web : page d'envoi de vidéo + analyse en tâche de fond + résultats."""
import os
import tempfile
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

import pipeline

app = FastAPI(title="Sprint Engine")

JOBS = {}
RUN_LOCK = threading.Semaphore(1)   # une analyse à la fois (mémoire limitée)
MAX_UPLOAD_MB = 150


def _worker(job_id, path, height_m, fps, max_s, playback, pauses):
    job = JOBS[job_id]
    with RUN_LOCK:
        try:
            job["status"] = "running"

            def prog(frac):
                job["progress"] = min(99, int(100 * frac))

            job["result"] = pipeline.run_pipeline(path, height_m=height_m, fps_override=fps, max_seconds=max_s,
                                                  progress=prog, out_video=job["video"], playback=playback, pauses=pauses)
            job["status"], job["progress"] = "done", 100
        except Exception as e:  # noqa
            traceback.print_exc()
            job["status"], job["error"] = "error", str(e)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), height_m: float = Form(1.75),
                  fps: str = Form(""), max_seconds: float = Form(6.0),
                  playback: str = Form("slow"), pauses: str = Form("1")):
    if not 1.2 <= height_m <= 2.3:
        raise HTTPException(400, "Taille hors limites (1,2 à 2,3 m).")
    if not 1 <= max_seconds <= 15:
        raise HTTPException(400, "Durée analysée : entre 1 et 15 secondes.")
    fps_val = None
    if fps.strip():
        try:
            fps_val = float(fps.replace(",", "."))
        except ValueError:
            raise HTTPException(400, "Cadence invalide.")
        if not 24 <= fps_val <= 1000:
            raise HTTPException(400, "Cadence entre 24 et 1000 images/s.")

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    too_big = False
    with os.fdopen(fd, "wb") as f:
        while True:
            chunk = await video.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                too_big = True
                break
            f.write(chunk)
    if too_big:
        os.remove(path)
        raise HTTPException(413, f"Vidéo trop lourde (max {MAX_UPLOAD_MB} Mo). Raccourcis-la.")

    if playback not in ("slow", "real"):
        playback = "slow"
    job_id = uuid.uuid4().hex[:10]
    vfd, vpath = tempfile.mkstemp(suffix=".mp4")
    os.close(vfd)
    JOBS[job_id] = {"status": "queued", "progress": 0, "created": time.time(), "video": vpath}
    for old in sorted(JOBS, key=lambda k: JOBS[k]["created"])[:-8]:   # on garde les 8 dernières
        gone = JOBS.pop(old, None)
        if gone and gone.get("video"):
            try:
                os.remove(gone["video"])
            except OSError:
                pass
    threading.Thread(target=_worker, args=(job_id, path, height_m, fps_val, max_seconds, playback, pauses == "1"), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Analyse introuvable (le serveur a peut-être redémarré).")
    return {k: v for k, v in job.items() if k not in ("created", "video")}


def _iter_file(path, start, end, chunk=1 << 20):
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            data = f.read(min(chunk, left))
            if not data:
                break
            left -= len(data)
            yield data


@app.get("/api/video/{job_id}")
def video(job_id: str, request: Request):
    job = JOBS.get(job_id)
    path = job.get("video") if job else None
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise HTTPException(404, "Vidéo annotée introuvable.")
    size = os.path.getsize(path)
    rng = pipeline.parse_range(request.headers.get("range"), size)
    if rng == "invalid":
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    if rng:
        start, end = rng
        return StreamingResponse(_iter_file(path, start, end), status_code=206, media_type="video/mp4",
                                 headers={"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}",
                                          "Content-Length": str(end - start + 1)})
    return StreamingResponse(_iter_file(path, 0, size - 1), media_type="video/mp4",
                             headers={"Accept-Ranges": "bytes", "Content-Length": str(size)})


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sprint Engine</title>
<style>
:root{--bg:#fff;--fg:#14181f;--mut:#667085;--card:#f4f6f9;--acc:#0b5fff;--bad:#b42318;--warn:#b54708}
@media(prefers-color-scheme:dark){:root{--bg:#0f1319;--fg:#e8ecf2;--mut:#98a2b3;--card:#1a212b;--acc:#6ea0ff;--bad:#f97066;--warn:#fdb022}}
*{box-sizing:border-box}body{margin:0;font:16px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg);padding:16px;max-width:640px;margin:auto}
h1{font-size:22px;margin:4px 0 2px}p.s{color:var(--mut);margin:0 0 14px}
label{display:block;font-weight:600;margin:12px 0 4px}small{color:var(--mut);font-weight:400}
input,select{width:100%;padding:11px;border-radius:10px;border:1px solid var(--mut);background:var(--card);color:var(--fg);font-size:16px}
button{width:100%;margin-top:16px;padding:14px;border:0;border-radius:12px;background:var(--acc);color:#fff;font-size:17px;font-weight:700}
button:disabled{opacity:.5}
.card{background:var(--card);border-radius:12px;padding:12px;margin:12px 0}
.bar{height:10px;background:var(--card);border-radius:6px;overflow:hidden;margin-top:8px}.bar i{display:block;height:100%;width:0;background:var(--acc);transition:width .3s}
table{width:100%;border-collapse:collapse}td,th{padding:6px 4px;text-align:left;border-bottom:1px solid rgba(128,128,128,.25);font-size:15px}
.big{font-size:20px;font-weight:700}.err{color:var(--bad);font-weight:600}.warn{color:var(--warn);margin:4px 0;font-size:14px}
.wrap{overflow-x:auto}.st{border-left:4px solid var(--acc);padding:8px 10px;margin:8px 0;background:var(--bg);border-radius:8px}.st b{font-size:16px}.st .n{color:var(--mut);font-size:14px}.st .c{margin-top:4px;font-size:14px}.btn2{width:auto;margin:6px 6px 0 0;padding:8px 12px;font-size:14px;border-radius:10px}
</style></head><body>
<h1>Sprint Engine</h1><p class="s">Analyse de la foulée à partir d'une vidéo, caméra stable ou non.</p>
<div class="card">
<label>Vidéo <small>(profil, athlète entier visible, 4 à 8 s)</small></label>
<input id="v" type="file" accept="video/*">
<label>Taille de l'athlète en mètres</label>
<input id="h" type="number" step="0.01" value="1.75" inputmode="decimal">
<label>Cadence réelle de tournage <small>(images/s, laisse vide si le fichier est correct)</small></label>
<input id="f" type="text" placeholder="ex. 240" inputmode="decimal">
<label>Durée analysée max <small>(secondes)</small></label>
<input id="m" type="number" step="1" value="6" inputmode="numeric">
<label>Pause à chaque foulée <small>(fiche complète + commentaire)</small></label>
<select id="pz"><option value="1">Oui, avec fiche détaillée</option><option value="0">Non, lecture continue</option></select>
<label>Lecture de la vidéo annotée</label>
<select id="pb2"><option value="slow">Ralenti (plus lisible)</option><option value="real">Vitesse réelle</option></select>
<button id="go">Analyser</button>
</div>
<div id="st"></div><div id="out"></div>
<script>
const $=id=>document.getElementById(id);
function cell(m){return m?`${m.valeur} <small>± ${m.incertitude}</small>`:'-'}
function show(r,id){
 const m=r.metriques,q=r.qualite;let h='';
 if(r.video_annotee){h+=`<div class="card"><video controls playsinline loop style="width:100%;border-radius:10px" src="/api/video/${id}"></video>
 <button class="btn2" id="vx">🔊 Voix du commentateur : désactivée</button>
 <a href="/api/video/${id}" download="analyse_sprint.mp4" style="color:var(--acc)">Télécharger la vidéo</a></div>`}
 if(m.erreur){h+=`<div class="card err">${m.erreur}</div>`}
 else{h+=`<div class="card"><table>
 <tr><th>Mesure</th><th>Valeur ± incertitude</th></tr>
 <tr><td>Temps de contact (ms)</td><td class="big">${cell(m.temps_contact_ms)}</td></tr>
 <tr><td>Temps de vol (ms)</td><td class="big">${cell(m.temps_vol_ms)}</td></tr>
 <tr><td>Fréquence (Hz)</td><td class="big">${cell(m.frequence_foulee_hz)}</td></tr>
 <tr><td>Longueur de foulée (m)</td><td class="big">${cell(m.longueur_foulee_m)}</td></tr>
 <tr><td>Vitesse moyenne (m/s)</td><td class="big">${cell(m.vitesse_moyenne_ms)}</td></tr>
 <tr><td>Inclinaison du tronc (°)</td><td class="big">${cell(m.inclinaison_tronc_deg)}</td></tr>
 <tr><td>Angle du genou à l'attaque (°)</td><td class="big">${cell(m.angle_genou_a_l_attaque_deg)}</td></tr>
 </table></div>`}
 h+=`<div class="card"><b>Qualité</b><br>Athlète détecté : ${q.athlete_detecte_pct} % · Pieds visibles : ${q.pieds_visibles_pct} % · Stabilisation : ${q.stabilisation_fiabilite_pct} %<br>
 Sens : ${r.sens_course} · Échelle : ${r.echelle_px_par_m} px/m · ${r.images_analysees} images à ${r.fps_utilise} i/s`;
 if(r.vitesse_max_hanches_ms)h+=`<br>Vitesse max des hanches : ${r.vitesse_max_hanches_ms} m/s`;
 h+='</div>';
 h+='<div class="card">'+r.avertissements.map(w=>`<div class="warn">⚠ ${w}</div>`).join('')+'</div>';
 if(r.appuis.length){h+=`<div class="card wrap"><b>Appuis détectés</b><table><tr><th>Pied</th><th>Début s</th><th>Durée ms</th><th>x m</th></tr>`+
  r.appuis.map(a=>`<tr><td>${a.pied}</td><td>${a.debut_s}</td><td>${a.duree_ms}</td><td>${a.x_m}</td></tr>`).join('')+'</table></div>'}
 if(r.bilan&&r.bilan.length){h+='<div class="card"><b>Bilan</b>'+r.bilan.map(b=>`<div class="c">• ${b}</div>`).join('')+'</div>'}
 if(r.foulees&&r.foulees.length){h+='<div class="card"><b>Foulée par foulée</b>'+r.foulees.map((s,i)=>{
  const f=(v,u)=>v==null?'--':v+u;
  return `<div class="st"><b>Foulée ${s.n} · pied ${s.pied==='G'?'gauche':'droit'}</b>
  <div class="n">contact ${f(s.contact_ms,' ms')} (±${s.incertitude_ms}) · vol ${f(s.vol_ms,' ms')} · longueur ${f(s.longueur_m,' m')} · ${f(s.frequence_hz,' Hz')} · ${f(s.vitesse_ms,' m/s')}<br>
  attaque ${f(s.attaque_m,' m')} · genou ${f(s.genou_attaque_deg,'°')} (min ${f(s.genou_flexion_max_deg,'°')}) · tronc ${f(s.tronc_deg,'°')}</div>
  <div class="c">${s.commentaires.join(' ')}</div>
  ${s.t_out!=null?`<button class="btn2" onclick="seek(${s.t_out})">Voir dans la vidéo</button>`:''}</div>`}).join('')+'</div>'}
 $('out').innerHTML=h;
 CH=(r.foulees||[]).filter(s=>s.t_out!=null&&s.pause_s>0);
 const vid=document.querySelector('#out video'),vx=$('vx');
 if(vid&&vx){
  vx.onclick=()=>{voiceOn=!voiceOn;vx.textContent='🔊 Voix du commentateur : '+(voiceOn?'activée':'désactivée');if(voiceOn)speak('Voix activée.');else if('speechSynthesis' in window)speechSynthesis.cancel()};
  vid.addEventListener('timeupdate',()=>{if(!voiceOn)return;const t=vid.currentTime;const i=CH.findIndex(c=>t>=c.t_out&&t<c.t_out+c.pause_s);if(i>=0&&i!==lastCh){lastCh=i;speak(CH[i].voix)}if(i<0)lastCh=-1});
  vid.addEventListener('pause',()=>{if('speechSynthesis' in window)speechSynthesis.cancel()});
  vid.addEventListener('seeking',()=>{lastCh=-1;if('speechSynthesis' in window)speechSynthesis.cancel()});
 }
}
let voiceOn=false,lastCh=-1,CH=[];
function speak(t){if(!('speechSynthesis' in window))return;speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(t);u.lang='fr-FR';u.rate=1.0;speechSynthesis.speak(u)}
function seek(t){const v=document.querySelector('#out video');if(v){v.currentTime=t;v.play();v.scrollIntoView({behavior:'smooth',block:'center'})}}
$('go').onclick=async()=>{
 const file=$('v').files[0];if(!file){$('st').innerHTML='<div class="err">Choisis une vidéo.</div>';return}
 $('go').disabled=true;$('out').innerHTML='';
 $('st').innerHTML='<div class="card">Envoi de la vidéo…<div class="bar"><i id="pb"></i></div></div>';
 const fd=new FormData();fd.append('video',file);fd.append('height_m',$('h').value);fd.append('fps',$('f').value);fd.append('max_seconds',$('m').value);fd.append('playback',$('pb2').value);fd.append('pauses',$('pz').value);
 try{
  const r=await fetch('/api/analyze',{method:'POST',body:fd});
  if(!r.ok){const e=await r.json().catch(()=>({detail:'Erreur '+r.status}));throw new Error(e.detail||('Erreur '+r.status))}
  const {job_id}=await r.json();
  while(true){
   await new Promise(r=>setTimeout(r,2000));
   const s=await (await fetch('/api/status/'+job_id)).json();
   if(s.detail)throw new Error(s.detail);
   $('st').innerHTML=`<div class="card">${s.status==='queued'?'En attente…':'Analyse et montage en cours… '+s.progress+' %'}<div class="bar"><i style="width:${s.progress}%"></i></div></div>`;
   if(s.status==='done'){$('st').innerHTML='';show(s.result,job_id);break}
   if(s.status==='error')throw new Error(s.error);
  }
 }catch(e){$('st').innerHTML=`<div class="card err">${e.message}</div>`}
 $('go').disabled=false;
};
</script></body></html>
"""
