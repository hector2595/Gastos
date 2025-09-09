import os
import streamlit as st
from supabase import create_client

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

sb = None
if SUPABASE_URL and SUPABASE_KEY:
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# App: Gastos en Pareja — Streamlit (100% Python)
# Autor: HÈTORj

# Requisitos (requirements.txt):
# streamlit
# pandas
# python-dateutil
# supabase
# psycopg2-binary  # solo si usas Postgres directo; para Supabase no es obligatorio

# Secrets en Streamlit (Settings → Secrets):
# SUPABASE_URL = "https://TU-PROYECTO.supabase.co"
# SUPABASE_KEY = "TU-ANON-KEY"

import os
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import List, Optional, Dict

import pandas as pd
import streamlit as st

# ---------------------------
# Supabase (opcional)
# ---------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_sb = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client, Client
        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        _sb = None

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Gastos en Pareja", page_icon="💸", layout="wide")
st.title("💸 Gastos en Pareja — Control diario")

# Archivo CSV (fallback si no hay Supabase)
DATA_DIR = ".data"
CSV_PATH = os.path.join(DATA_DIR, "expenses.csv")
CAT_CSV_PATH = os.path.join(DATA_DIR, "categories.csv")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_CATS = [
    "Comida trabajo","Comida casa","Gasolina","Juego de fútbol","Panapass","Internet casa"
]

# ---------------------------
# Helpers de persistencia
# ---------------------------

def ensure_local_files():
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=[
            "id","date","category","description","amount","who","method","merchant","notes","ts","created_at"
        ]).to_csv(CSV_PATH, index=False)
    if not os.path.exists(CAT_CSV_PATH):
        pd.DataFrame({"name": DEFAULT_CATS, "active": [True]*len(DEFAULT_CATS)}).to_csv(CAT_CSV_PATH, index=False)

def load_categories() -> List[str]:
    if _sb:
        res = _sb.table("categories").select("name, active").eq("active", True).order("name").execute()
        data = res.data or []
        return [r["name"] for r in data]
    else:
        ensure_local_files()
        dfc = pd.read_csv(CAT_CSV_PATH)
        dfc = dfc[dfc["active"] == True]
        return sorted(dfc["name"].dropna().unique().tolist())

def add_category(name: str):
    name = name.strip()
    if not name:
        return
    if _sb:
        _sb.table("categories").insert({"name": name, "active": True}).execute()
    else:
        ensure_local_files()
        dfc = pd.read_csv(CAT_CSV_PATH)
        if name not in dfc["name"].values:
            dfc = pd.concat([dfc, pd.DataFrame({"name": [name], "active": [True]})])
            dfc.to_csv(CAT_CSV_PATH, index=False)

def soft_delete_category(name: str):
    if _sb:
        _sb.table("categories").update({"active": False}).eq("name", name).execute()
    else:
        dfc = pd.read_csv(CAT_CSV_PATH)
        dfc.loc[dfc["name"] == name, "active"] = False
        dfc.to_csv(CAT_CSV_PATH, index=False)

def insert_expense(payload: Dict):
    payload = dict(payload)
    payload.setdefault("ts", datetime.utcnow().isoformat())
    payload.setdefault("created_at", datetime.utcnow().isoformat())
    if _sb:
        _sb.table("expenses").insert(payload).execute()
    else:
        ensure_local_files()
        df = pd.read_csv(CSV_PATH)
        payload.setdefault("id", pd.util.hash_pandas_object(pd.DataFrame([payload])).astype(str).iloc[0])
        df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

def query_expenses(start: date, end: date, who: Optional[List[str]] = None, cats: Optional[List[str]] = None) -> pd.DataFrame:
    if _sb:
        q = _sb.table("expenses").select("*").gte("date", str(start)).lte("date", str(end))
        if who:
            q = q.in_("who", who)
        if cats:
            q = q.in_("category", cats)
        res = q.order("date").execute()
        df = pd.DataFrame(res.data or [])
        return df
    else:
        ensure_local_files()
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
        else:
            df = pd.DataFrame(columns=["date","category","description","amount","who","method","merchant","notes","ts","created_at"])
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        mask = (df["date"] >= start) & (df["date"] <= end)
        if who:
            mask &= df["who"].isin(who)
        if cats:
            mask &= df["category"].isin(cats)
        return df.loc[mask].copy()

# ---------------------------
# Sidebar: filtros & configuración
# ---------------------------
with st.sidebar:
    st.header("Filtros")
    hoy = date.today()
    inicio = st.date_input("Desde", hoy.replace(day=1))
    fin = st.date_input("Hasta", hoy)

    who_opts = ["Héctor","Erika","Ambos"]
    who_sel = st.multiselect("Quién pagó", who_opts)

    cats = load_categories()
    cat_sel = st.multiselect("Categorías", cats)

    st.markdown("---")
    st.subheader("Categorías")
    new_cat = st.text_input("Nueva categoría")
    ccol1, ccol2 = st.columns([1,1])
    with ccol1:
        if st.button("➕ Agregar", use_container_width=True):
            add_category(new_cat)
            st.experimental_rerun()
    with ccol2:
        del_cat = st.selectbox("Desactivar", ["(seleccionar)"] + cats)
        if st.button("🗑️ Desactivar", use_container_width=True) and del_cat != "(seleccionar)":
            soft_delete_category(del_cat)
            st.experimental_rerun()

# ---------------------------
# Formulario de alta
# ---------------------------
st.subheader("Agregar gasto")
col1, col2, col3, col4 = st.columns([1,1,2,2])
with col1:
    f_fecha = st.date_input("Fecha", date.today(), key="fecha_form")
    f_quien = st.selectbox("Quién pagó", ["Héctor","Erika","Ambos"], key="quien_form")
with col2:
    f_monto = st.number_input("Monto", min_value=0.00, step=0.25, format="%.2f", key="monto_form")
    cats_for_form = load_categories()  # asegurar refresco si se agregan
    f_cat = st.selectbox("Categoría", options=(cats_for_form or ["Comida trabajo"]), key="cat_form")
with col3:
    f_desc = st.text_input("Descripción", key="desc_form", placeholder="Ej. almuerzo, peaje, tanque lleno…")
    f_method = st.text_input("Método de pago", key="method_form", placeholder="Efectivo, Visa, Yappy…")
with col4:
    f_merchant = st.text_input("Comercio (opcional)", key="merchant_form")
    f_notes = st.text_input("Notas (opcional)", key="notes_form")

if st.button("Guardar ✅", type="primary"):
    if f_monto and f_cat:
        insert_expense({
            "date": str(f_fecha),
            "category": f_cat,
            "description": f_desc,
            "amount": float(f_monto),
            "who": f_quien,
            "method": f_method,
            "merchant": f_merchant,
            "notes": f_notes
        })
        st.success("Gasto guardado")
    else:
        st.error("Completa al menos Monto y Categoría")

# ---------------------------
# Consulta y Resumen
# ---------------------------
df = query_expenses(inicio, fin, who_sel or None, cat_sel or None)

st.subheader("Resumen del período")
if df is None or df.empty:
    st.info("Sin datos en el rango y filtros seleccionados.")
else:
    # Normalización
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    total = float(df["amount"].sum())
    dias = max(1, (fin - inicio).days + 1)
    prom_diario = total / dias

    # ======== NUEVAS MÉTRICAS SEPARADAS ========
    # Totales generales
    top1, top2 = st.columns(2)
    top1.metric("Total período", f"${total:,.2f}")
    top2.metric("Promedio diario", f"${prom_diario:,.2f}")

    # Por persona (separado, sin mezclar 'Ambos')
    p1, p2, p3 = st.columns(3)
    sum_h = float(df.loc[df["who"] == "Héctor", "amount"].sum()) if "who" in df.columns else 0.0
    sum_e = float(df.loc[df["who"] == "Erika",  "amount"].sum()) if "who" in df.columns else 0.0
    sum_a = float(df.loc[df["who"] == "Ambos",  "amount"].sum()) if "who" in df.columns else 0.0

    p1.metric("Pagado por Héctor", f"${sum_h:,.2f}")
    p2.metric("Pagado por Erika",  f"${sum_e:,.2f}")
    p3.metric("Pagado por Ambos",  f"${sum_a:,.2f}")

    # (Opcional) Totales ajustados repartiendo 'Ambos' 50/50
    # Descomenta si quieres ver esta fila extra:
    # adj1, adj2 = st.columns(2)
    # adj_h = sum_h + (sum_a / 2.0)
    # adj_e = sum_e + (sum_a / 2.0)
    # adj1.metric("Héctor (con 50% de 'Ambos')", f"${adj_h:,.2f}")
    # adj2.metric("Erika (con 50% de 'Ambos')",  f"${adj_e:,.2f}")
    # ======== FIN CAMBIO ========

    # Gráficos
    st.markdown("### Gasto por categoría")
    cat_sum = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    st.bar_chart(cat_sum)

    st.markdown("### Gasto por día")
    by_day = df.groupby("date")["amount"].sum()
    st.line_chart(by_day)

    st.markdown("### Detalle de transacciones")
    show_cols = [c for c in ["date","category","description","amount","who","method","merchant","notes"] if c in df.columns]
    st.dataframe(df[show_cols].sort_values(by=["date","amount"], ascending=[True, False]), use_container_width=True)

    # Exportación
    st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"), "gastos_filtrados.csv", "text/csv")

    # ---------------------------
    # Detección simple de "fugas"
    # ---------------------------
    st.markdown("### 🔎 Fugas potenciales (gastos inusuales)")
    # Comparar gasto diario del período vs. promedio móvil de 30 días (necesita histórico)
    full_hist = query_expenses(date(2000,1,1), fin)
    alerts = []
    if full_hist is not None and not full_hist.empty:
        full_hist["date"] = pd.to_datetime(full_hist["date"], errors="coerce").dt.date
        daily = full_hist.groupby("date")["amount"].sum().reset_index()
        daily = daily.sort_values("date")
        # promedio móvil de 30 días
        s = pd.Series(daily["amount"].values)
        roll_mean = s.rolling(window=30, min_periods=7).mean()
        roll_std = s.rolling(window=30, min_periods=7).std().fillna(0.0)
        z = (s - roll_mean) / (roll_std.replace(0, pd.NA))
        # puntos con z>2 en el rango actual
        daily["z"] = z
        current_mask = (daily["date"] >= inicio) & (daily["date"] <= fin)
        spikes = daily.loc[current_mask & (daily["z"] > 2)]
        if not spikes.empty:
            for _, r in spikes.iterrows():
                alerts.append(f"Día {r['date']}: gasto total ${r['amount']:.2f} > tendencia 30d (z={r['z']:.2f})")

        # Categorías fuera de patrón (top 3 con mayor desviación vs su media mensual)
        df_m = full_hist.copy()
        df_m["ym"] = pd.to_datetime(df_m["date"]).dt.to_period('M')
        cat_month = df_m.groupby(["ym","category"])['amount'].sum().reset_index()
        # periodo actual
        ym_cur = pd.to_datetime(pd.Series([inicio])).dt.to_period('M').iloc[0]
        cm_cur = cat_month.loc[cat_month['ym'] == ym_cur]
        if not cm_cur.empty:
            base = cat_month.loc[cat_month['ym'] != ym_cur].groupby("category")['amount'].mean().reset_index(name='mean_prev')
            joined = pd.merge(cm_cur, base, on='category', how='left').fillna({'mean_prev': 0.0})
            joined['delta'] = joined['amount'] - joined['mean_prev']
            joined = joined.sort_values('delta', ascending=False).head(3)
            for _, r in joined.iterrows():
                if r['delta'] > 0:
                    alerts.append(f"Categoría {r['category']} supera promedio histórico mensual por ${r['delta']:.2f}")

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("Sin fugas evidentes en el período.")

