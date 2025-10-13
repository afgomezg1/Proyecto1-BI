const { useState, useEffect, useRef } = React;

/* ===================== Helpers ===================== */
const API_BASE = ""; // mismo origen. Si separas front y API: "http://localhost:8000"

const api = {
  async predict(payload) {
    const res = await fetch(API_BASE + "/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json().catch(()=> ({}));
    if (!res.ok) throw new Error(data?.detail || data?.error || res.statusText);
    return data;
  },
  async retrain(payload) {
    const res = await fetch(API_BASE + "/retrain", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json().catch(()=> ({}));
    if (!res.ok) throw new Error(data?.detail || data?.error || res.statusText);
    return data;
  },
};

/* ===== CSV parsing: UNA sola columna por archivo =====
   - No dependemos de coma/;/".
   - Forzamos un delimitador "imposible" para que cada línea sea una celda.
   - Si la primera línea coincide con `expectedHeader` ("texto" o "label"), se ignora.
*/
async function parseSingleColumnCSV(file, expectedHeader) {
  const readAsText = (f) =>
    new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = () => res(r.result);
      r.onerror = rej;
      r.readAsText(f, "utf-8");
    });

  const raw = await readAsText(file);

  // Delimitador “imposible” (símbolo visible del Unit Separator)
  const impossibleDelim = "\u241F";

  const parsed = Papa.parse(raw, {
    delimiter: impossibleDelim,
    header: false,
    skipEmptyLines: "greedy",
  });

  const lines = (parsed.data || [])
    .map((arr) => (Array.isArray(arr) ? arr[0] : arr))
    .map((v) => String(v ?? "").trim())
    .filter((v) => v !== "");

  if (!lines.length) return [];

  // Ignora encabezado si coincide (case-insensitive)
  const first = (lines[0] || "").toLowerCase();
  if (expectedHeader && first === String(expectedHeader).toLowerCase()) {
    return lines.slice(1);
  }
  return lines;
}

/* Vista previa simple de hasta 10 filas (una columna) */
function SmallPreview({ header, values }) {
  if (!values?.length) return null;
  return (
    <div className="table-responsive">
      <table className="table table-sm table-striped">
        <thead><tr><th>{header}</th></tr></thead>
        <tbody>
          {values.slice(0,10).map((v,i)=>(<tr key={i}><td>{String(v)}</td></tr>))}
        </tbody>
      </table>
      <div className="small-muted">Mostrando {Math.min(values.length,10)} de {values.length} filas.</div>
    </div>
  );
}

function DropZone({ onFiles, label=".csv" }) {
  const ref = useRef(null);
  useEffect(()=>{
    const el = ref.current;
    if (!el) return;
    const onOver  = e => { e.preventDefault(); el.classList.add("dragover"); };
    const onLeave = e => { e.preventDefault(); el.classList.remove("dragover"); };
    const onDrop  = e => {
      e.preventDefault();
      el.classList.remove("dragover");
      const files = [...(e.dataTransfer?.files||[])];
      if (files.length) onFiles(files);
    };
    el.addEventListener("dragover", onOver);
    el.addEventListener("dragleave", onLeave);
    el.addEventListener("drop", onDrop);
    return ()=> {
      el.removeEventListener("dragover", onOver);
      el.removeEventListener("dragleave", onLeave);
      el.removeEventListener("drop", onDrop);
    };
  }, [onFiles]);
  return (
    <div ref={ref} className="dropzone">
      Arrastra y suelta tu archivo <strong>{label}</strong> aquí, o usa el botón “Seleccionar archivo”.
    </div>
  );
}

/* ============== UI de Probabilidades (solo barras, en %) ============== */
function ProbBarsPercent({ probs }) {
  const entries = Object.entries(probs || {}).sort((a,b)=>b[1]-a[1]);
  if (!entries.length) return <span className="text-muted small">Sin probabilidades</span>;
  return (
    <div>
      {entries.map(([k,v])=>{
        const pct = Math.max(0, Math.min(100, (v || 0) * 100));
        const label = `${pct.toFixed(2)}%`;
        return (
          <div key={k} className="mb-2">
            <div className="d-flex justify-content-between">
              <small className="text-muted">ODS {k}</small>
              <small className="text-muted">{label}</small>
            </div>
            <div className="prog"><div style={{width:`${pct}%`}}></div></div>
          </div>
        );
      })}
    </div>
  );
}

function PredictionCard({ index, label, probs }) {
  return (
    <div className="card shadow-sm mb-3">
      <div className="card-body">
        <h6 className="mb-2">#{index} — Predicción: <span className="badge bg-primary">ODS {label}</span></h6>
        <ProbBarsPercent probs={probs} />
      </div>
    </div>
  );
}

/* ===================== Indicadores de proceso ===================== */
function Processing({ show, done }) {
  if (!show && !done) return null;
  if (show) {
    return (
      <div className="alert alert-info d-flex align-items-center py-2 my-2" role="alert">
        <div className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></div>
        <div><strong>Procesando…</strong> Por favor espera.</div>
      </div>
    );
  }
  if (done) {
    return (
      <div className="alert alert-success py-2 my-2" role="alert">
        <strong>Procesado ✓</strong>
      </div>
    );
  }
  return null;
}

/* ===================== Pantallas ===================== */
function Home() {
  return (
    <main className="container">
      <div className="row">
        <div className="col-md-8 mx-auto text-center">
          <h1 className="mb-3">Sistema de Clasificación de Textos en ODS</h1>
          <p className="lead">Clasifica tus textos por ODS y mejora el modelo con nuevos ejemplos etiquetados.</p>

          <div className="row mt-4 g-4">
            <div className="col-md-6">
              <div className="card h-100 shadow-sm">
                <div className="card-body d-flex flex-column">
                  <h5 className="card-title">Clasificar Texto</h5>
                  <p className="card-text flex-grow-1">Ingresa texto (formulario) o sube un <code>.csv</code> con una sola columna <code>texto</code>.</p>
                  <a href="#/predict" className="btn btn-primary mt-2">Ir a Clasificar</a>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100 shadow-sm">
                <div className="card-body d-flex flex-column">
                  <h5 className="card-title">Reentrenar Modelo</h5>
                  <p className="card-text flex-grow-1">Usa el formulario o sube dos <code>.csv</code>: uno con <code>texto</code> y otro con <code>label</code> (solo 1, 3 o 4).</p>
                  <a href="#/retrain" className="btn btn-primary mt-2">Ir a Reentrenar</a>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}

/* ---------- Predict (form 1 campo + CSV 1 columna) ---------- */
function Predict() {
  const [rows, setRows] = useState([{ id: 1, texto:"" }]);
  const [csvTextos, setCsvTextos] = useState([]);   // valores de la columna 'texto'
  const [processing, setProcessing] = useState(false);
  const [processed, setProcessed] = useState(false);
  const [result, setResult] = useState(null);

  const addRow = () => setRows(prev => [...prev, { id: Date.now(), texto:"" }]);
  const clearRow = (id) => setRows(prev => prev.map(r => r.id===id ? {...r, texto:""} : r));
  const removeRow = (id) => setRows(prev => prev.length>1 ? prev.filter(r=>r.id!==id): prev);
  const setField = (id, v) => setRows(prev => prev.map(r => r.id===id ? {...r, texto:v} : r));

  const clearCSV = (inputId) => {
    const input = document.getElementById(inputId);
    if (input) input.value = "";
    setCsvTextos([]);
  };
  const clearResults = () => { setResult(null); setProcessed(false); };

  const buildTextosFromRows = () =>
    rows.map(r => String(r.texto||"").trim()).filter(Boolean);

  const predictManual = async () => {
    const textos = buildTextosFromRows();
    if (!textos.length) return alert("Agrega al menos un texto.");
    setProcessing(true); setProcessed(false);
    try{
      const data = await api.predict({ textos });
      setResult(data);
      setProcessed(true);
    }catch(e){ alert(e.message || e); }
    finally { setProcessing(false); }
  };

  const predictCSV = async () => {
    if (!csvTextos.length) return alert("Adjunta o arrastra un CSV válido con columna 'texto'.");
    setProcessing(true); setProcessed(false);
    try{
      const data = await api.predict({ textos: csvTextos });
      setResult(data);
      setProcessed(true);
    }catch(e){ alert(e.message || e); }
    finally { setProcessing(false); }
  };

  const onCsvTextosInput = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const values = await parseSingleColumnCSV(f, "texto");
      setCsvTextos(values);
    } catch (err) {
      alert(err.message || err);
    }
  };
  const onCsvTextosDrop = async (files) => {
    const f = files?.[0];
    if (!f) return;
    try {
      const values = await parseSingleColumnCSV(f, "texto");
      setCsvTextos(values);
    } catch (err) {
      alert(err.message || err);
    }
  };

  const preds = result?.predictions || [];
  const probas = result?.probabilities || [];

  return (
    <main className="container">
      <h2 className="mb-3">Clasificar Textos</h2>

      <div className="row g-4">
        <div className="col-lg-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <h5>Formulario (múltiples filas)</h5>
              <p className="small-muted">Completa el <strong>texto</strong> por cada fila.</p>
              {rows.map((r,idx)=>(
                <div className="border rounded p-3 mb-2 bg-white" key={r.id}>
                  <div className="d-flex justify-content-between align-items-center">
                    <strong>Texto #{idx+1}</strong>
                    <div className="d-flex gap-2">
                      <button className="btn btn-sm btn-outline-secondary" onClick={()=>clearRow(r.id)}>Limpiar</button>
                      <button className="btn btn-sm btn-outline-danger" onClick={()=>removeRow(r.id)} disabled={rows.length===1}>Eliminar</button>
                    </div>
                  </div>
                  <textarea className="form-control mt-2" rows="2" placeholder="Texto" value={r.texto} onChange={e=>setField(r.id,e.target.value)}></textarea>
                </div>
              ))}
              <div className="d-flex gap-2 align-items-center">
                <button className="btn btn-outline-secondary mt-2" onClick={addRow}>Agregar fila</button>
                <button className="btn btn-primary mt-2" onClick={predictManual} disabled={processing}>Clasificar</button>
              </div>
              <Processing show={processing} done={processed} />
            </div>
          </div>
        </div>

        <div className="col-lg-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <div className="d-flex justify-content-between align-items-center">
                <h5 className="mb-0">CSV (1 columna: <code>texto</code>)</h5>
                <button className="btn btn-sm btn-outline-secondary" onClick={()=>clearCSV("predict-csv")}>Quitar archivo</button>
              </div>
              <input id="predict-csv" type="file" accept=".csv" className="form-control mb-2" onChange={onCsvTextosInput} />
              <DropZone onFiles={onCsvTextosDrop} label=".csv" />
              <div className="d-flex gap-2 mt-2">
                <button className="btn btn-outline-secondary mt-2" onClick={()=>{ /* preview ya visible */ }}>Previsualizar</button>
                <button className="btn btn-primary mt-2" onClick={predictCSV} disabled={processing}>Clasificar CSV</button>
              </div>
              <div className="mt-3"><SmallPreview header="texto" values={csvTextos} /></div>
              <Processing show={processing} done={processed} />
            </div>
          </div>
        </div>
      </div>

      <section className="mt-4">
        <div className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">Resultados</h5>
          <button className="btn btn-sm btn-outline-secondary" onClick={()=>{ setResult(null); setProcessed(false); }}>Limpiar resultados</button>
        </div>
        {preds.length===0 ? (
          <p className="text-muted mt-2">Sin resultados.</p>
        ) : (
          preds.map((y,i)=><PredictionCard key={i} index={i+1} label={y} probs={probas[i]||{}} />)
        )}
      </section>
    </main>
  );
}

/* ---------- Retrain (form 2 campos + DOS CSV: texto y label) ---------- */
function Retrain() {
  const [rows, setRows] = useState([{ id:1, texto:"", label:"" }]);

  const [csvTextos, setCsvTextos] = useState([]); // desde CSV de textos (columna 'texto')
  const [csvLabels, setCsvLabels] = useState([]); // desde CSV de labels (columna 'label')

  const [processing, setProcessing] = useState(false);
  const [processed, setProcessed] = useState(false);
  const [result, setResult] = useState(null);

  const allowedLabels = new Set([1,3,4]);

  const addRow = () => setRows(prev => [...prev, { id: Date.now(), texto:"", label:"" }]);
  const clearRow = (id) => setRows(prev => prev.map(r => r.id===id ? {...r, texto:"", label:""} : r));
  const removeRow = (id) => setRows(prev => prev.length>1 ? prev.filter(r=>r.id!==id): prev);
  const setField = (id, k, v) => setRows(prev => prev.map(r => r.id===id ? {...r, [k]:v} : r));

  const clearCSV = (inputId, which) => {
    const input = document.getElementById(inputId);
    if (input) input.value = "";
    if (which === "textos") setCsvTextos([]);
    if (which === "labels") setCsvLabels([]);
  };
  const clearResults = () => { setResult(null); setProcessed(false); };

  const buildPayloadFromRows = () => {
    const textos = [], labels = [];
    for (const r of rows) {
      const t = String(r.texto||"").trim();
      const l = parseInt(String(r.label||"").trim(), 10);
      if (t && !Number.isNaN(l) && allowedLabels.has(l)) { textos.push(t); labels.push(l); }
    }
    return { textos, labels };
  };

  const retrainManual = async () => {
    const payload = buildPayloadFromRows();
    if (!payload.textos.length || payload.textos.length !== payload.labels.length)
      return alert("Completa texto y una etiqueta válida (1, 3 o 4) en cada fila.");
    setProcessing(true); setProcessed(false);
    try{
      const data = await api.retrain(payload);
      setResult(data);
      setProcessed(true);
    }catch(e){ alert(e.message || e); }
    finally { setProcessing(false); }
  };

  const retrainCSV = async () => {
    if (!csvTextos.length || !csvLabels.length)
      return alert("Adjunta ambos CSV: uno con 'texto' y otro con 'label'.");
    if (csvTextos.length !== csvLabels.length)
      return alert(`Los CSV no coinciden en longitud: textos=${csvTextos.length}, labels=${csvLabels.length}.`);

    const labelsParsed = csvLabels.map(v => parseInt(String(v).trim(), 10));
    if (labelsParsed.some(v => Number.isNaN(v) || !allowedLabels.has(v)))
      return alert("El CSV de 'label' solo admite valores 1, 3 o 4.");

    setProcessing(true); setProcessed(false);
    try{
      const data = await api.retrain({ textos: csvTextos, labels: labelsParsed });
      setResult(data);
      setProcessed(true);
    }catch(e){ alert(e.message || e); }
    finally { setProcessing(false); }
  };

  // Inputs CSV
  const onCsvTextosInput = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const values = await parseSingleColumnCSV(f, "texto");
      setCsvTextos(values);
    } catch (err) {
      alert(err.message || err);
    }
  };
  const onCsvTextosDrop = async (files) => {
    const f = files?.[0];
    if (!f) return;
    try {
      const values = await parseSingleColumnCSV(f, "texto");
      setCsvTextos(values);
    } catch (err) {
      alert(err.message || err);
    }
  };

  const onCsvLabelsInput = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try {
      const values = await parseSingleColumnCSV(f, "label");
      setCsvLabels(values);
    } catch (err) {
      alert(err.message || err);
    }
  };
  const onCsvLabelsDrop = async (files) => {
    const f = files?.[0];
    if (!f) return;
    try {
      const values = await parseSingleColumnCSV(f, "label");
      setCsvLabels(values);
    } catch (err) {
      alert(err.message || err);
    }
  };

  // Utilidad: fecha desde nb_pipeline_v2_YYYYMMDD_HHMMSS.joblib
  function formatModelInfo(path) {
    if (!path) return { name: null, dateStr: null };
    const name = path.split("/").pop();
    const m = name.match(/nb_pipeline_v2_(\d{8})_(\d{6})\.joblib$/);
    if (!m) return { name, dateStr: null };
    const [_, ymd, hms] = m;
    const yyyy = ymd.slice(0,4), mm = ymd.slice(4,6), dd = ymd.slice(6,8);
    const hh = hms.slice(0,2), mi = hms.slice(2,4), ss = hms.slice(4,6);
    return { name, dateStr: `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}` };
  }

  function RetrainSummary({ data }) {
    if (!data) return null;
    const status = data.status ?? "ok";
    const { name: modelName, dateStr } = formatModelInfo(data.new_model_path);
    const sizes = data.sizes || {};
    const before = data.metrics_before || {};
    const after = data.metrics_after || {};

    const rows = [
      ["accuracy", before.accuracy, after.accuracy],
      ["balanced_acc", before.balanced_acc, after.balanced_acc],
      ["precision_macro", before.precision_macro, after.precision_macro],
      ["recall_macro", before.recall_macro, after.recall_macro],
      ["f1_macro", before.f1_macro, after.f1_macro],
    ];

    const fmtPct = (v)=> (v==null ? "N/A" : `${(v*100).toFixed(2)} %`);

    return (
      <div className="card shadow-sm">
        <div className="card-body">
          <div className="d-flex justify-content-between align-items-center mb-2">
            <h5 className="mb-0">Resultado del Reentrenamiento</h5>
            <span className={`badge ${status==='ok' ? 'text-bg-success' : 'text-bg-danger'}`}>{status?.toUpperCase()}</span>
          </div>

          <div className="mb-2">
            <div><strong>Modelo nuevo:</strong> <code>{modelName || "-"}</code></div>
            {dateStr && <div><strong>Fecha del modelo:</strong> {dateStr}</div>}
            <div><strong>Ruta:</strong> {data.new_model_path}</div>
          </div>

          <div className="mb-3">
            <div><strong>Histórico (antes):</strong> {sizes.n_history_before ?? "N/A"}</div>
            <div><strong>Nuevos:</strong> {sizes.n_new ?? "N/A"}</div>
            <div><strong>Histórico (después):</strong> {sizes.n_history_after ?? "N/A"}</div>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-striped align-middle">
              <thead>
                <tr><th>Métrica</th><th>Antes</th><th>Después</th><th>Δ</th></tr>
              </thead>
              <tbody>
                {rows.map(([name, b, a])=>{
                  const d = ( (a??0) - (b??0) ) * 100;
                  const cls = (a==null || b==null) ? 'text-muted' : d>0 ? 'text-success' : d<0 ? 'text-danger' : 'text-muted';
                  const arrow = (a==null || b==null) ? '→' : d>0 ? '↑' : d<0 ? '↓' : '→';
                  return (
                    <tr key={name}>
                      <td><code>{name}</code></td>
                      <td>{fmtPct(b)}</td>
                      <td>{fmtPct(a)}</td>
                      <td className={cls}>{arrow} {isNaN(d) ? "N/A" : `${Math.abs(d).toFixed(2)} pp`}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  }

  return (
    <main className="container">
      <h2 className="mb-3">Reentrenar Modelo</h2>

      <div className="row g-4">
        <div className="col-lg-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <h5>Formulario (múltiples filas)</h5>
              <p className="small-muted">Completa <strong>texto</strong> y su <strong>etiqueta</strong> (1, 3 o 4).</p>
              {rows.map((r,idx)=>(
                <div className="border rounded p-3 mb-2 bg-white" key={r.id}>
                  <div className="d-flex justify-content-between align-items-center">
                    <strong>Ejemplo #{idx+1}</strong>
                    <div className="d-flex gap-2">
                      <button className="btn btn-sm btn-outline-secondary" onClick={()=>clearRow(r.id)}>Limpiar</button>
                      <button className="btn btn-sm btn-outline-danger" onClick={()=>removeRow(r.id)} disabled={rows.length===1}>Eliminar</button>
                    </div>
                  </div>
                  <textarea className="form-control mt-2" rows="2" placeholder="Texto" value={r.texto} onChange={e=>setField(r.id,"texto",e.target.value)}></textarea>
                  <select className="form-select mt-2" value={r.label} onChange={e=>setField(r.id,"label",e.target.value)}>
                    <option value="">Seleccione etiqueta (1, 3 o 4)</option>
                    <option value="1">1</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                  </select>
                </div>
              ))}
              <div className="d-flex gap-2 align-items-center">
                <button className="btn btn-outline-secondary mt-2" onClick={addRow}>Agregar fila</button>
                <button className="btn btn-primary mt-2" onClick={retrainManual} disabled={processing}>Reentrenar</button>
              </div>
              <Processing show={processing} done={processed} />
            </div>
          </div>
        </div>

        <div className="col-lg-6">
          <div className="card shadow-sm">
            <div className="card-body">
              <h5 className="mb-2">CSV por separado</h5>
              <p className="small-muted mb-1">• <strong>Textos:</strong> CSV con una sola columna <code>texto</code>.</p>
              <p className="small-muted">• <strong>Etiquetas:</strong> CSV con una sola columna <code>label</code> (solo 1, 3 o 4). Ambos archivos deben tener la <strong>misma cantidad</strong> de filas.</p>

              <div className="mb-3">
                <div className="d-flex justify-content-between align-items-center">
                  <h6 className="mb-0">CSV — Textos</h6>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=>clearCSV("retrain-csv-textos","textos")}>Quitar archivo</button>
                </div>
                <input id="retrain-csv-textos" type="file" accept=".csv" className="form-control mb-2" onChange={onCsvTextosInput} />
                <DropZone onFiles={onCsvTextosDrop} label=".csv (texto)" />
                <div className="mt-2"><SmallPreview header="texto" values={csvTextos} /></div>
              </div>

              <div className="mb-3">
                <div className="d-flex justify-content-between align-items-center">
                  <h6 className="mb-0">CSV — Labels</h6>
                  <button className="btn btn-sm btn-outline-secondary" onClick={()=>clearCSV("retrain-csv-labels","labels")}>Quitar archivo</button>
                </div>
                <input id="retrain-csv-labels" type="file" accept=".csv" className="form-control mb-2" onChange={onCsvLabelsInput} />
                <DropZone onFiles={onCsvLabelsDrop} label=".csv (label)" />
                <div className="mt-2"><SmallPreview header="label" values={csvLabels} /></div>
              </div>

              <div className="d-flex gap-2 mt-2">
                <button className="btn btn-primary mt-2" onClick={retrainCSV} disabled={processing}>Reentrenar con CSV</button>
              </div>

              <Processing show={processing} done={processed} />
            </div>
          </div>
        </div>
      </div>

      <section className="mt-4">
        <div className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">Resultados</h5>
          <button className="btn btn-sm btn-outline-secondary" onClick={()=>{ setResult(null); setProcessed(false); }}>Limpiar resultados</button>
        </div>
        {!result ? (
          <p className="text-muted mt-2">Sin resultados.</p>
        ) : (
          <RetrainSummary data={result} />
        )}
      </section>
    </main>
  );
}

/* ===================== Router simple (hash) ===================== */
function App() {
  const [route, setRoute] = useState(window.location.hash || "#/");
  useEffect(() => {
    const onHash = () => setRoute(window.location.hash || "#/");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  if (route.startsWith("#/predict")) return <Predict />;
  if (route.startsWith("#/retrain")) return <Retrain />;
  return <Home />;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
