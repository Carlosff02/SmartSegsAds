"""
================================================================================
 SmartSeg Ads — Panel de Machine Learning para Segmentación de Prospectos
 Árbol de Decisión con selección de variables (X) y comparación Train/Test
 Digital Boost Agency | Innovación y Transformación Digital - UTP
================================================================================

Cómo ejecutar:
    pip install pandas numpy scikit-learn matplotlib openpyxl
    python smartseg_ads_gui.py

Requiere Tkinter (incluido por defecto en la instalación estándar de Python
para Windows y macOS; en Linux: sudo apt install python3-tk)

Formato del Excel esperado — columnas obligatorias:
    clics | ctr | tiempo_sesion | paginas_vistas | dispositivo | edad_segmento | convirtio
    dispositivo  : 1=Móvil, 2=Escritorio, 3=Tablet
    edad_segmento: 1=18-24, 2=25-34, 3=35-44, 4=45+
    convirtio    : 0=No convirtió, 1=Convirtió

Novedades de esta versión (pensadas para el curso Innovación y
Transformación Digital - UTP):
    1) El usuario puede elegir qué variables independientes (X) usar para
       entrenar el árbol de decisión, desde la pestaña "Configuración".
    2) Se puede comparar el desempeño en Entrenamiento vs. Prueba (pestaña
       "Entrenamiento vs Prueba"), con métricas, matrices de confusión y
       una lectura pedagógica sobre sobreajuste (overfitting).
    3) Interfaz reorganizada con identidad institucional UTP y textos que
       conectan cada pantalla con los conceptos del curso.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ==============================================================================
#  PALETA DE COLORES — identidad UTP (rojo institucional) sobre tema oscuro
# ==============================================================================
class Theme:
    BG0            = "#15161A"
    BG1            = "#1B1D22"
    BG2            = "#21242B"
    BORDER         = "#2E3138"
    BORDER_STRONG  = "#3D4149"
    TEXT_PRIMARY   = "#E7E8EA"
    TEXT_SECONDARY = "#A9ACB4"
    TEXT_MUTED     = "#75787F"
    ACCENT         = "#D93A46"   # rojo UTP
    ACCENT_BG      = "#2C1619"
    GOLD           = "#D9A441"  # dorado institucional (acento secundario)
    GOLD_BG        = "#2B2415"
    SUCCESS        = "#34C38F"
    SUCCESS_BG     = "#16261F"
    WARNING        = "#F0A93B"
    WARNING_BG     = "#2B2415"
    DANGER         = "#E0556F"
    DANGER_BG      = "#2B1A20"
    BLUE           = "#378ADD"
    GREEN          = "#1D9E75"
    ORANGE         = "#D85A30"
    PURPLE         = "#7F77DD"
    FONT_FAMILY    = "Segoe UI"
    FONT_MONO      = "Consolas"


ALL_FEATURES = ["clics", "ctr", "tiempo_sesion", "paginas_vistas", "dispositivo", "edad_segmento"]

FEATURE_LABELS = {
    "clics": "Clics", "ctr": "CTR", "tiempo_sesion": "Tiempo sesión",
    "paginas_vistas": "Páginas vistas", "dispositivo": "Dispositivo",
    "edad_segmento": "Segmento etario",
}

FEATURE_DESCRIPTIONS = {
    "clics":          "Número de clics en el anuncio · variable cuantitativa discreta",
    "ctr":            "Click-Through Rate: efectividad del anuncio (%) · cuantitativa continua",
    "tiempo_sesion":  "Minutos de permanencia del usuario · cuantitativa continua",
    "paginas_vistas": "Páginas visitadas en la sesión · cuantitativa discreta",
    "dispositivo":    "Móvil / Escritorio / Tablet · variable categórica",
    "edad_segmento":  "Rango etario del usuario · variable categórica",
}

DISPOSITIVO_MAP = {1: "Móvil", 2: "Escritorio", 3: "Tablet"}
EDAD_MAP        = {1: "18–24 años", 2: "25–34 años", 3: "35–44 años", 4: "45+ años"}

COLUMNAS_REQUERIDAS = ALL_FEATURES + ["convirtio"]


# ==============================================================================
#  MOTOR DE MACHINE LEARNING
# ==============================================================================
class SmartSegModel:
    """
    Encapsula todo el flujo de ML: carga/generación de datos, limpieza,
    balanceo, entrenamiento del árbol de decisión con variables (X)
    seleccionables por el usuario, y comparación Entrenamiento vs Prueba.
    """

    def __init__(self, seed: int = 42):
        self.seed        = seed

        self.df_raw      = None
        self.df_bal      = None
        self.scaler      = None
        self.clf         = None

        # Variables (X) actualmente usadas para entrenar el modelo
        self.features_used = list(ALL_FEATURES)
        self.max_depth      = 5
        self.test_size       = 0.20

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred_train = self.y_pred_test = None

        self.train_metrics = {}
        self.test_metrics  = {}
        self.metrics        = {}   # alias de test_metrics (compatibilidad)

        self.importances    = None
        self.rules_text      = ""
        self.n_outliers_removed = 0
        self.fuente          = "simulado"   # "simulado" | ruta del archivo Excel
        self.n_raw           = 0

    # ── A. Datos simulados (fallback / demo) ─────────────────────────────────
    def generar_dataset_simulado(self):
        np.random.seed(self.seed)
        n = 1200
        clics          = np.random.randint(0, 20, n)
        ctr            = np.round(np.random.uniform(0.5, 7.0, n), 2)
        tiempo_sesion  = np.round(np.random.uniform(0.1, 15.0, n), 2)
        paginas_vistas = np.random.randint(1, 12, n)
        dispositivo    = np.random.choice([1, 2, 3], n, p=[0.50, 0.40, 0.10])
        edad_segmento  = np.random.choice([1, 2, 3, 4], n, p=[0.30, 0.35, 0.25, 0.10])
        prob = (
            0.30*(ctr/7.0) + 0.25*(tiempo_sesion/15.0) + 0.20*(paginas_vistas/12.0)
            + 0.10*(clics/20.0) + 0.10*np.where(dispositivo==2,1,0)
            + 0.05*(edad_segmento/4.0) + np.random.normal(0, 0.08, n)
        )
        convirtio = (prob > 0.40).astype(int)
        self.df_raw = pd.DataFrame({
            "clics": clics, "ctr": ctr, "tiempo_sesion": tiempo_sesion,
            "paginas_vistas": paginas_vistas, "dispositivo": dispositivo,
            "edad_segmento": edad_segmento, "convirtio": convirtio,
        })
        self.fuente = "simulado"
        self.n_raw  = len(self.df_raw)
        return self.df_raw

    # ── B. Carga desde Excel ──────────────────────────────────────────────────
    def cargar_excel(self, ruta: str):
        """
        Lee el Excel, valida columnas y tipos, devuelve (ok:bool, mensaje:str).
        Si ok=True, self.df_raw queda listo para entrenar.
        """
        try:
            df = pd.read_excel(ruta, engine="openpyxl")
        except Exception as e:
            return False, f"No se pudo abrir el archivo:\n{e}"

        faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
        if faltantes:
            return False, (
                f"El archivo no tiene las columnas requeridas:\n"
                f"{', '.join(faltantes)}\n\n"
                f"Columnas esperadas:\n{', '.join(COLUMNAS_REQUERIDAS)}"
            )

        df = df[COLUMNAS_REQUERIDAS].copy()

        for col in COLUMNAS_REQUERIDAS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        disp_vals = df["dispositivo"].dropna().unique()
        invalid_disp = [v for v in disp_vals if v not in [1, 2, 3]]
        if invalid_disp:
            return False, (
                f"Valores inválidos en 'dispositivo': {invalid_disp}\n"
                f"Permitidos: 1=Móvil, 2=Escritorio, 3=Tablet"
            )
        edad_vals = df["edad_segmento"].dropna().unique()
        invalid_edad = [v for v in edad_vals if v not in [1, 2, 3, 4]]
        if invalid_edad:
            return False, (
                f"Valores inválidos en 'edad_segmento': {invalid_edad}\n"
                f"Permitidos: 1=18-24, 2=25-34, 3=35-44, 4=45+"
            )
        conv_vals = df["convirtio"].dropna().unique()
        invalid_conv = [v for v in conv_vals if v not in [0, 1]]
        if invalid_conv:
            return False, (
                f"Valores inválidos en 'convirtio': {invalid_conv}\n"
                f"Permitidos: 0=No convirtió, 1=Convirtió"
            )

        if len(df) < 20:
            return False, f"El archivo tiene solo {len(df)} filas. Se requieren al menos 20."

        self.df_raw = df.reset_index(drop=True)
        self.fuente = ruta
        self.n_raw  = len(self.df_raw)
        return True, f"Archivo cargado: {len(df)} registros."

    # ── C. Limpieza y balanceo (usa solo las X seleccionadas) ────────────────
    def limpiar_y_balancear(self):
        df = self.df_raw.copy()
        antes = len(df)
        df = df[df["tiempo_sesion"] <= 120].dropna()
        self.n_outliers_removed = antes - len(df)

        n_min = df["convirtio"].value_counts().min()
        df_0 = df[df["convirtio"] == 0].sample(n=n_min, random_state=self.seed)
        df_1 = df[df["convirtio"] == 1].sample(n=n_min, random_state=self.seed)
        self.df_bal = (
            pd.concat([df_0, df_1])
            .sample(frac=1, random_state=self.seed)
            .reset_index(drop=True)
        )

        X = self.df_bal[self.features_used]
        y = self.df_bal["convirtio"]
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    # ── D. Entrenamiento con comparación Train vs Test ───────────────────────
    def entrenar(self, max_depth: int = 5, features=None, test_size: float = 0.20):
        if not features:
            features = list(ALL_FEATURES)
        # Se conserva el orden canónico de ALL_FEATURES para mayor claridad
        self.features_used = [f for f in ALL_FEATURES if f in features]
        self.max_depth = max_depth
        self.test_size = test_size

        X_scaled, y = self.limpiar_y_balancear()
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        self.clf = DecisionTreeClassifier(
            max_depth=max_depth, criterion="gini", random_state=self.seed
        )
        self.clf.fit(X_train, y_train)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.rules_text = export_text(self.clf, feature_names=self.features_used)

        self.y_pred_train = self.clf.predict(X_train)
        self.y_pred_test  = self.clf.predict(X_test)

        self.train_metrics = self._calc_metrics(y_train, self.y_pred_train)
        self.test_metrics  = self._calc_metrics(y_test, self.y_pred_test)
        self.metrics = self.test_metrics  # compatibilidad con vistas existentes

        self.importances = pd.Series(
            self.clf.feature_importances_, index=self.features_used
        ).sort_values(ascending=False)
        return self.test_metrics

    @staticmethod
    def _calc_metrics(y_true, y_pred):
        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "n": len(y_true),
        }

    # ── E. Predicción de prospecto nuevo (usa solo las X seleccionadas) ──────
    def predecir(self, clics, ctr, tiempo_sesion, paginas_vistas, dispositivo, edad_segmento):
        valores = {
            "clics": clics, "ctr": ctr, "tiempo_sesion": tiempo_sesion,
            "paginas_vistas": paginas_vistas, "dispositivo": dispositivo,
            "edad_segmento": edad_segmento,
        }
        nuevo = pd.DataFrame([valores])[self.features_used]
        nuevo_scaled = self.scaler.transform(nuevo)
        pred  = self.clf.predict(nuevo_scaled)[0]
        proba = self.clf.predict_proba(nuevo_scaled)[0]
        return int(pred), proba


# ==============================================================================
#  GENERADOR DE EXCEL DE EJEMPLO
# ==============================================================================
def generar_excel_ejemplo(ruta: str):
    """Crea un Excel con 20 filas de datos de muestra para que el usuario
    lo use como plantilla y complete con sus datos reales (hasta 1200 filas)."""
    np.random.seed(0)
    n = 20
    data = {
        "clics":          np.random.randint(0, 20, n),
        "ctr":            np.round(np.random.uniform(0.5, 7.0, n), 2),
        "tiempo_sesion":  np.round(np.random.uniform(0.1, 15.0, n), 2),
        "paginas_vistas": np.random.randint(1, 12, n),
        "dispositivo":    np.random.choice([1, 2, 3], n),
        "edad_segmento":  np.random.choice([1, 2, 3, 4], n),
        "convirtio":      np.random.choice([0, 1], n),
    }
    df = pd.DataFrame(data)

    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Prospectos")
        ref = pd.DataFrame({
            "Columna":      COLUMNAS_REQUERIDAS,
            "Tipo":         ["Entero","Decimal","Decimal","Entero","Entero","Entero","Entero (0/1)"],
            "Descripción":  [
                "Número de clics en el anuncio (0–19)",
                "Click-Through Rate en % (0.5–7.0)",
                "Tiempo de sesión en minutos (0.1–15.0)",
                "Páginas vistas durante la sesión (1–11)",
                "Dispositivo: 1=Móvil · 2=Escritorio · 3=Tablet",
                "Edad: 1=18-24 · 2=25-34 · 3=35-44 · 4=45+",
                "Variable objetivo: 1=Convirtió · 0=No convirtió",
            ],
        })
        ref.to_excel(writer, index=False, sheet_name="Referencia")


# ==============================================================================
#  WIDGETS AUXILIARES
# ==============================================================================
def make_card(parent, **kwargs):
    options = {"bg": Theme.BG2, "highlightbackground": Theme.BORDER,
               "highlightthickness": 1, "bd": 0}
    options.update(kwargs)
    return tk.Frame(parent, **options)


def make_badge(parent, text, kind="muted"):
    palette = {
        "success": (Theme.SUCCESS_BG, Theme.SUCCESS),
        "warning": (Theme.WARNING_BG, Theme.WARNING),
        "accent":  (Theme.ACCENT_BG,  Theme.ACCENT),
        "gold":    (Theme.GOLD_BG,    Theme.GOLD),
        "danger":  (Theme.DANGER_BG,  Theme.DANGER),
        "muted":   (Theme.BG1,        Theme.TEXT_SECONDARY),
    }
    bg, fg = palette.get(kind, palette["muted"])
    return tk.Label(parent, text=text, bg=bg, fg=fg,
                    font=(Theme.FONT_FAMILY, 9, "bold"), padx=8, pady=3)


def progress_row(parent, label, pct, color):
    row = tk.Frame(parent, bg=Theme.BG2)
    row.pack(fill="x", pady=3)
    tk.Label(row, text=label, bg=Theme.BG2, fg=Theme.TEXT_PRIMARY,
             font=(Theme.FONT_FAMILY, 9), width=18, anchor="w").pack(side="left")

    bar = tk.Canvas(row, height=8, bg=Theme.BG1, highlightthickness=0)
    bar.pack(side="left", padx=8, fill="x", expand=True)

    def draw(event=None):
        bar.delete("all")
        w = bar.winfo_width() or 300
        bar.create_rectangle(0, 0, w, 8, fill=Theme.BG1, outline="")
        fw = max(2, int(w * min(max(pct, 0), 100) / 100))
        bar.create_rectangle(0, 0, fw, 8, fill=color, outline="")

    bar.bind("<Configure>", draw)
    tk.Label(row, text=f"{pct:.1f}%", bg=Theme.BG2, fg=Theme.TEXT_MUTED,
             font=(Theme.FONT_FAMILY, 9), width=6, anchor="e").pack(side="left")


def paired_bar_row(parent, label, val_train, val_test, fmt="{:.1f}%"):
    """Fila que compara visualmente un valor de entrenamiento vs prueba."""
    row = tk.Frame(parent, bg=Theme.BG2)
    row.pack(fill="x", pady=5)
    tk.Label(row, text=label, bg=Theme.BG2, fg=Theme.TEXT_PRIMARY,
             font=(Theme.FONT_FAMILY, 9), width=14, anchor="w").pack(side="left")

    bar = tk.Canvas(row, height=16, bg=Theme.BG1, highlightthickness=0)
    bar.pack(side="left", padx=8, fill="x", expand=True)

    def draw(event=None):
        bar.delete("all")
        w = bar.winfo_width() or 300
        h = 16
        bar.create_rectangle(0, 0, w, h, fill=Theme.BG1, outline="")
        fw_train = max(2, int(w * min(max(val_train, 0), 100) / 100))
        fw_test  = max(2, int(w * min(max(val_test, 0), 100) / 100))
        bar.create_rectangle(0, 0, fw_train, h//2 - 1, fill=Theme.BLUE, outline="")
        bar.create_rectangle(0, h//2 + 1, fw_test, h, fill=Theme.ACCENT, outline="")

    bar.bind("<Configure>", draw)
    tk.Label(row, text=f"{fmt.format(val_train)} / {fmt.format(val_test)}",
             bg=Theme.BG2, fg=Theme.TEXT_MUTED, font=(Theme.FONT_FAMILY, 9),
             width=13, anchor="e").pack(side="left")


# ==============================================================================
#  APLICACIÓN PRINCIPAL
# ==============================================================================
class SmartSegApp(tk.Tk):
    TABS = ["Panel", "Configuración", "Segmentos", "Predicción",
            "Entrenamiento vs Prueba", "Reglas del árbol"]

    def __init__(self):
        super().__init__()
        self.title("SmartSeg Ads — Innovación y Transformación Digital · UTP")
        self.geometry("1220x800")
        self.minsize(1040, 680)
        self.configure(bg=Theme.BG0)

        self.model      = SmartSegModel()
        self.active_tab = tk.StringVar(value="Panel")
        self.nav_buttons = {}

        # Variables de configuración (deben existir antes de entrenar)
        self.feature_vars = {f: tk.BooleanVar(value=True) for f in ALL_FEATURES}
        self.var_max_depth = tk.IntVar(value=5)
        self.var_test_size = tk.DoubleVar(value=0.20)
        self.feature_count_label = None

        self._build_topbar()
        self._build_nav()

        self.main_container = tk.Frame(self, bg=Theme.BG0)
        self.main_container.pack(fill="both", expand=True)

        self.views = {}
        self._build_status_bar()
        self._entrenar_inicial_simulado()
        self._build_views()
        self._show_tab("Panel")

    # ─────────────────────────────────────────────── TOPBAR ──────────────────
    def _build_topbar(self):
        bar = tk.Frame(self, bg=Theme.BG2, height=58)
        bar.pack(fill="x", side="top")

        left = tk.Frame(bar, bg=Theme.BG2)
        left.pack(side="left", padx=18, pady=8)
        tk.Label(left, text="UTP", bg=Theme.ACCENT, fg="#FFFFFF",
                 font=(Theme.FONT_FAMILY, 11, "bold"), padx=8, pady=4
                 ).grid(row=0, column=0, rowspan=2, padx=(0, 10))
        tk.Label(left, text="SmartSeg Ads", bg=Theme.BG2, fg=Theme.TEXT_PRIMARY,
                 font=(Theme.FONT_FAMILY, 13, "bold")
                 ).grid(row=0, column=1, sticky="w")
        tk.Label(left, text="Innovación y Transformación Digital · Digital Boost Agency",
                 bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                 font=(Theme.FONT_FAMILY, 8)
                 ).grid(row=1, column=1, sticky="w")

        right = tk.Frame(bar, bg=Theme.BG2)
        right.pack(side="right", padx=18, pady=14)
        self.status_badge = make_badge(right, "  ●  Cargando...  ", "warning")
        self.status_badge.pack(side="right", padx=(8, 0))

        btn_frame = tk.Frame(bar, bg=Theme.BG2)
        btn_frame.pack(side="right", padx=(0, 12), pady=14)

        btn_load = tk.Label(
            btn_frame, text="📂  Cargar Excel", bg=Theme.ACCENT_BG, fg=Theme.ACCENT,
            font=(Theme.FONT_FAMILY, 9, "bold"), padx=10, pady=4, cursor="hand2"
        )
        btn_load.pack(side="left", padx=(0, 6))
        btn_load.bind("<Button-1>", lambda e: self._cargar_excel())

        btn_demo = tk.Label(
            btn_frame, text="💾  Descargar plantilla", bg=Theme.BG1, fg=Theme.TEXT_SECONDARY,
            font=(Theme.FONT_FAMILY, 9), padx=10, pady=4, cursor="hand2"
        )
        btn_demo.pack(side="left")
        btn_demo.bind("<Button-1>", lambda e: self._descargar_plantilla())

    # ─────────────────────────────────────────────── NAV ─────────────────────
    def _build_nav(self):
        nav = tk.Frame(self, bg=Theme.BG1, height=40)
        nav.pack(fill="x", side="top")
        inner = tk.Frame(nav, bg=Theme.BG1)
        inner.pack(side="left", padx=14)
        for tab in self.TABS:
            btn = tk.Label(inner, text=tab, bg=Theme.BG1, fg=Theme.TEXT_SECONDARY,
                           font=(Theme.FONT_FAMILY, 10), padx=12, pady=9, cursor="hand2")
            btn.pack(side="left")
            btn.bind("<Button-1>", lambda e, t=tab: self._show_tab(t))
            btn.bind("<Enter>",    lambda e, b=btn, t=tab: self._on_nav_hover(b, t, True))
            btn.bind("<Leave>",    lambda e, b=btn, t=tab: self._on_nav_hover(b, t, False))
            self.nav_buttons[tab] = btn

    def _on_nav_hover(self, btn, tab, entering):
        if tab == self.active_tab.get():
            return
        btn.configure(fg=Theme.TEXT_PRIMARY if entering else Theme.TEXT_SECONDARY)

    def _show_tab(self, tab):
        self.active_tab.set(tab)
        for t, btn in self.nav_buttons.items():
            if t == tab:
                btn.configure(fg=Theme.ACCENT, font=(Theme.FONT_FAMILY, 10, "bold"))
            else:
                btn.configure(fg=Theme.TEXT_SECONDARY, font=(Theme.FONT_FAMILY, 10))
        for frame in self.views.values():
            frame.pack_forget()
        if tab in self.views:
            self.views[tab].pack(fill="both", expand=True)

    # ─────────────────────────────────────────────── STATUS BAR ──────────────
    def _build_status_bar(self):
        bar = tk.Frame(self, bg=Theme.BG1, height=24)
        bar.pack(fill="x", side="bottom")
        self.status_text = tk.Label(
            bar, text="Listo.", bg=Theme.BG1, fg=Theme.TEXT_MUTED,
            font=(Theme.FONT_FAMILY, 8), anchor="w", padx=12
        )
        self.status_text.pack(side="left", fill="x")
        self.fuente_label = tk.Label(
            bar, text="Fuente: dataset simulado", bg=Theme.BG1, fg=Theme.TEXT_MUTED,
            font=(Theme.FONT_FAMILY, 8), anchor="e", padx=12
        )
        self.fuente_label.pack(side="right")

    # ─────────────────────────────────── CARGA Y ENTRENAMIENTO ───────────────
    def _selected_features(self):
        sel = [f for f in ALL_FEATURES if self.feature_vars[f].get()]
        return sel if sel else list(ALL_FEATURES)

    def _entrenar_inicial_simulado(self):
        self.model.generar_dataset_simulado()
        self.model.entrenar(
            max_depth=self.var_max_depth.get(),
            features=self._selected_features(),
            test_size=self.var_test_size.get(),
        )
        self._actualizar_badge_modelo()
        self._actualizar_fuente_label()

    def _actualizar_badge_modelo(self):
        acc = self.model.test_metrics.get("accuracy", 0)
        n_feat = len(self.model.features_used)
        self.status_badge.configure(
            text=f"  ●  Modelo activo · {acc*100:.1f}% acc (test) · {n_feat} variables  ",
            bg=Theme.SUCCESS_BG, fg=Theme.SUCCESS
        )

    def _actualizar_fuente_label(self):
        fuente = self.model.fuente
        if fuente == "simulado":
            txt = f"Fuente: dataset simulado · {self.model.n_raw} registros"
        else:
            txt = f"Fuente: {os.path.basename(fuente)} · {self.model.n_raw} registros"
        self.fuente_label.configure(text=txt)

    def _cargar_excel(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo Excel de prospectos",
            filetypes=[("Archivos Excel", "*.xlsx *.xls"), ("Todos", "*.*")]
        )
        if not ruta:
            return

        self.status_badge.configure(
            text="  ●  Cargando Excel...  ", bg=Theme.WARNING_BG, fg=Theme.WARNING
        )
        self.update_idletasks()

        ok, msg = self.model.cargar_excel(ruta)
        if not ok:
            messagebox.showerror("Error al cargar Excel", msg)
            self._actualizar_badge_modelo()
            return

        self.status_badge.configure(
            text="  ●  Entrenando...  ", bg=Theme.WARNING_BG, fg=Theme.WARNING
        )
        self.update_idletasks()
        try:
            self.model.entrenar(
                max_depth=self.var_max_depth.get(),
                features=self._selected_features(),
                test_size=self.var_test_size.get(),
            )
        except Exception as e:
            messagebox.showerror("Error al entrenar", str(e))
            self._actualizar_badge_modelo()
            return

        self._actualizar_badge_modelo()
        self._actualizar_fuente_label()
        self.status_text.configure(
            text=f"Excel cargado: {msg} | accuracy test={self.model.test_metrics['accuracy']*100:.1f}%"
        )
        self._refresh_views()

    def _descargar_plantilla(self):
        ruta = filedialog.asksaveasfilename(
            title="Guardar plantilla Excel",
            defaultextension=".xlsx",
            initialfile="smartseg_plantilla.xlsx",
            filetypes=[("Archivo Excel", "*.xlsx")]
        )
        if not ruta:
            return
        try:
            generar_excel_ejemplo(ruta)
            messagebox.showinfo(
                "Plantilla guardada",
                f"Plantilla guardada en:\n{ruta}\n\n"
                "Contiene 20 filas de ejemplo y una hoja 'Referencia'\n"
                "con la descripción de cada columna.\n\n"
                "Reemplaza los datos con los de tu campaña real (hasta 1200\n"
                "filas) y usa '📂 Cargar Excel' para entrenar el modelo."
            )
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la plantilla:\n{e}")

    def _reentrenar_simulado(self):
        self.status_badge.configure(
            text="  ●  Reentrenando...  ", bg=Theme.WARNING_BG, fg=Theme.WARNING
        )
        self.update_idletasks()
        if self.model.fuente == "simulado":
            self.model.generar_dataset_simulado()
        self.model.entrenar(
            max_depth=self.var_max_depth.get(),
            features=self._selected_features(),
            test_size=self.var_test_size.get(),
        )
        self._actualizar_badge_modelo()
        self._actualizar_fuente_label()
        self.status_text.configure(
            text=f"Modelo reentrenado · accuracy test={self.model.test_metrics['accuracy']*100:.1f}%"
        )
        self._refresh_views()

    def _aplicar_configuracion(self):
        seleccionadas = [f for f in ALL_FEATURES if self.feature_vars[f].get()]
        if len(seleccionadas) == 0:
            messagebox.showwarning(
                "Selección inválida",
                "Debes seleccionar al menos una variable (X) para entrenar el modelo."
            )
            return
        self._reentrenar_simulado()
        if self.feature_count_label is not None:
            self.feature_count_label.configure(
                text=f"{len(seleccionadas)} de {len(ALL_FEATURES)} variables seleccionadas"
            )

    # ─────────────────────────────────────────────── VISTAS ──────────────────
    def _build_views(self):
        self.views["Panel"]                    = self._build_panel_view()
        self.views["Configuración"]             = self._build_configuracion_view()
        self.views["Segmentos"]                 = self._build_segmentos_view()
        self.views["Predicción"]                = self._build_prediccion_view()
        self.views["Entrenamiento vs Prueba"]    = self._build_comparacion_view()
        self.views["Reglas del árbol"]           = self._build_reglas_view()

    def _refresh_views(self):
        for tab in ("Panel", "Segmentos", "Entrenamiento vs Prueba", "Reglas del árbol"):
            if tab in self.views:
                self.views[tab].destroy()
        self.views["Panel"]                 = self._build_panel_view()
        self.views["Segmentos"]              = self._build_segmentos_view()
        self.views["Entrenamiento vs Prueba"] = self._build_comparacion_view()
        self.views["Reglas del árbol"]        = self._build_reglas_view()
        self._show_tab(self.active_tab.get())

    # ── PANEL ─────────────────────────────────────────────────────────────────
    def _build_panel_view(self):
        view   = tk.Frame(self.main_container, bg=Theme.BG0)
        m      = self.model
        scroll = self._scrollable(view)

        fuente_txt = ("Dataset simulado — carga tu Excel real (hasta 1200 filas) con '📂 Cargar Excel'"
                      if m.fuente == "simulado"
                      else f"Datos cargados desde: {os.path.basename(m.fuente)} "
                           f"· {m.n_raw} registros originales "
                           f"· {m.n_outliers_removed} outliers eliminados")
        alert_kind = Theme.WARNING_BG if m.fuente == "simulado" else Theme.ACCENT_BG
        alert_fg   = Theme.WARNING    if m.fuente == "simulado" else Theme.ACCENT
        alert = tk.Frame(scroll, bg=alert_kind, highlightbackground=alert_fg,
                         highlightthickness=1)
        alert.pack(fill="x", padx=16, pady=(14, 10))
        icon = "⚠" if m.fuente == "simulado" else "✓"
        tk.Label(alert, bg=alert_kind, fg=alert_fg, justify="left", anchor="w",
                 font=(Theme.FONT_FAMILY, 9),
                 text=f"{icon}  {fuente_txt}", wraplength=1120
                 ).pack(fill="x", padx=12, pady=10)

        tasa_conv = m.df_bal["convirtio"].mean() * 100
        n_total   = len(m.df_bal)
        kpi_wrap  = tk.Frame(scroll, bg=Theme.BG0)
        kpi_wrap.pack(fill="x", padx=16, pady=(0, 10))
        for i in range(4):
            kpi_wrap.columnconfigure(i, weight=1)
        kpis = [
            ("Tasa de conversión",      f"{tasa_conv:.1f}%",                          Theme.SUCCESS,      "Dataset balanceado"),
            ("Prospectos clasificados", f"{n_total:,}",                                Theme.TEXT_PRIMARY, "Dataset activo"),
            ("Exactitud (test)",        f"{m.test_metrics['accuracy']*100:.1f}%",     Theme.ACCENT,        f"{len(m.features_used)} variables · depth {m.max_depth}"),
            ("F1 Score (test)",         f"{m.test_metrics['f1']*100:.1f}%",           Theme.TEXT_PRIMARY, "Precisión + Recall"),
        ]
        for i, (lbl, val, col, sub) in enumerate(kpis):
            card = make_card(kpi_wrap)
            card.grid(row=0, column=i, sticky="nsew", padx=5)
            tk.Label(card, text=lbl, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 8)).pack(anchor="w", padx=12, pady=(10, 2))
            tk.Label(card, text=val, bg=Theme.BG2, fg=col,
                     font=(Theme.FONT_FAMILY, 19, "bold")).pack(anchor="w", padx=12)
            tk.Label(card, text=sub, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 8)).pack(anchor="w", padx=12, pady=(2, 10))

        grid2 = tk.Frame(scroll, bg=Theme.BG0)
        grid2.pack(fill="x", padx=16, pady=(0, 10))
        grid2.columnconfigure(0, weight=1)
        grid2.columnconfigure(1, weight=1)

        seg_card = make_card(grid2)
        seg_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(seg_card, text="👥  Distribución por dispositivo (dataset activo)",
                 bg=Theme.BG2, fg=Theme.TEXT_SECONDARY,
                 font=(Theme.FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=12, pady=(10, 8))
        seg_inner = tk.Frame(seg_card, bg=Theme.BG2)
        seg_inner.pack(fill="x", padx=12, pady=(0, 12))
        disp_pct = m.df_bal["dispositivo"].map(DISPOSITIVO_MAP).value_counts(normalize=True)*100
        cols_cycle = [Theme.BLUE, Theme.GREEN, Theme.ORANGE]
        for i, (seg, pct) in enumerate(disp_pct.items()):
            progress_row(seg_inner, seg, pct, cols_cycle[i % len(cols_cycle)])

        chart_card = make_card(grid2)
        chart_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(chart_card, text="📈  Métricas del modelo (conjunto de prueba)",
                 bg=Theme.BG2, fg=Theme.TEXT_SECONDARY,
                 font=(Theme.FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        self._embed_metrics_chart(chart_card)

        table_card = make_card(scroll)
        table_card.pack(fill="x", padx=16, pady=(0, 16))
        tk.Label(table_card, text="📋  Prospectos recientes — clasificación ML (muestra del dataset)",
                 bg=Theme.BG2, fg=Theme.TEXT_SECONDARY,
                 font=(Theme.FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=12, pady=(10, 8))
        self._build_prospect_table(table_card)
        return view

    def _embed_metrics_chart(self, parent):
        m   = self.model
        fig = Figure(figsize=(4.6, 2.6), dpi=100)
        fig.patch.set_facecolor(Theme.BG2)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(Theme.BG2)
        metricas = {"Accuracy": m.test_metrics["accuracy"], "Precision": m.test_metrics["precision"],
                    "Recall": m.test_metrics["recall"], "F1": m.test_metrics["f1"]}
        bars = ax.bar(list(metricas.keys()), list(metricas.values()),
                      color=[Theme.BLUE, Theme.GREEN, Theme.ORANGE, Theme.PURPLE], width=0.6)
        ax.set_ylim(0, 1.15)
        for bar, val in zip(bars, metricas.values()):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.03, f"{val*100:.1f}%",
                    ha="center", va="bottom", fontsize=8, color=Theme.TEXT_PRIMARY)
        ax.tick_params(colors=Theme.TEXT_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
        ax.axhline(y=0.80, color=Theme.TEXT_MUTED, linestyle="--", linewidth=0.7, alpha=0.6)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=parent).get_tk_widget().pack(
            fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_prospect_table(self, parent):
        cols    = ("prospecto","ctr","tiempo","dispositivo","prob","clasificacion")
        headers = ["Prospecto","CTR","Tiempo sesión","Dispositivo","Prob. conv.","Clasificación"]
        style   = ttk.Style()
        style.theme_use("clam")
        style.configure("SS.Treeview", background=Theme.BG2, foreground=Theme.TEXT_PRIMARY,
                        fieldbackground=Theme.BG2, rowheight=26, borderwidth=0,
                        font=(Theme.FONT_FAMILY, 9))
        style.configure("SS.Treeview.Heading", background=Theme.BG1,
                        foreground=Theme.TEXT_MUTED,
                        font=(Theme.FONT_FAMILY, 9, "bold"), borderwidth=0)
        style.map("SS.Treeview", background=[("selected", Theme.ACCENT_BG)])
        tree = ttk.Treeview(parent, columns=cols, show="headings",
                             style="SS.Treeview", height=6)
        for col, head in zip(cols, headers):
            tree.heading(col, text=head)
            tree.column(col, anchor="center", width=150)
        tree.pack(fill="x", padx=12, pady=(0, 12))

        m       = self.model
        muestra = m.df_bal.sample(n=min(8, len(m.df_bal)), random_state=7).reset_index(drop=True)
        probs   = m.clf.predict_proba(
            m.scaler.transform(muestra[m.features_used])
        )[:, 1] * 100
        for i, (_, row) in enumerate(muestra.iterrows()):
            prob = probs[i]
            if prob >= 65:   clasif, tag = "Convertirá",     "ok"
            elif prob >= 40: clasif, tag = "Probable",        "warn"
            else:            clasif, tag = "No convertirá",   "bad"
            tree.insert("", "end", values=(
                f"P-{1000+i}", f"{row['ctr']:.1f}%",
                f"{row['tiempo_sesion']:.1f} min",
                DISPOSITIVO_MAP.get(int(row["dispositivo"]), "—"),
                f"{prob:.0f}%", clasif,
            ), tags=(tag,))
        tree.tag_configure("ok",   foreground=Theme.SUCCESS)
        tree.tag_configure("warn", foreground=Theme.WARNING)
        tree.tag_configure("bad",  foreground=Theme.DANGER)

    # ── CONFIGURACIÓN (selección de variables X) ────────────────────────────
    def _build_configuracion_view(self):
        view   = tk.Frame(self.main_container, bg=Theme.BG0)
        scroll = self._scrollable(view)

        tk.Label(scroll, text="Configuración del modelo", bg=Theme.BG0,
                 fg=Theme.TEXT_PRIMARY, font=(Theme.FONT_FAMILY, 12, "bold")
                 ).pack(anchor="w", padx=16, pady=(14, 4))
        tk.Label(scroll,
                 text="Aplica los conceptos del curso: elige las variables independientes (X) "
                      "que alimentan al árbol de decisión y ajusta los parámetros del "
                      "entrenamiento antes de comparar resultados.",
                 bg=Theme.BG0, fg=Theme.TEXT_MUTED, font=(Theme.FONT_FAMILY, 8),
                 wraplength=1100, justify="left").pack(anchor="w", padx=16, pady=(0, 12))

        # ── Card: selección de variables ──
        feat_card = make_card(scroll)
        feat_card.pack(fill="x", padx=16, pady=(0, 12))
        head = tk.Frame(feat_card, bg=Theme.BG2)
        head.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(head, text="Variables predictoras (X) disponibles", bg=Theme.BG2,
                 fg=Theme.TEXT_SECONDARY, font=(Theme.FONT_FAMILY, 9, "bold")
                 ).pack(side="left")
        n_sel = sum(1 for f in ALL_FEATURES if self.feature_vars[f].get())
        self.feature_count_label = tk.Label(
            head, text=f"{n_sel} de {len(ALL_FEATURES)} variables seleccionadas",
            bg=Theme.BG2, fg=Theme.TEXT_MUTED, font=(Theme.FONT_FAMILY, 8)
        )
        self.feature_count_label.pack(side="right")

        feat_grid = tk.Frame(feat_card, bg=Theme.BG2)
        feat_grid.pack(fill="x", padx=14, pady=(0, 6))
        feat_grid.columnconfigure(0, weight=1)
        feat_grid.columnconfigure(1, weight=1)
        for i, feat in enumerate(ALL_FEATURES):
            r, c = divmod(i, 2)
            cell = tk.Frame(feat_grid, bg=Theme.BG1, highlightbackground=Theme.BORDER,
                             highlightthickness=1)
            cell.grid(row=r, column=c, sticky="ew", padx=6, pady=6)
            feat_grid.rowconfigure(r, weight=1)
            cb = tk.Checkbutton(
                cell, text=FEATURE_LABELS[feat], variable=self.feature_vars[feat],
                bg=Theme.BG1, fg=Theme.TEXT_PRIMARY, selectcolor=Theme.BG0,
                activebackground=Theme.BG1, activeforeground=Theme.TEXT_PRIMARY,
                font=(Theme.FONT_FAMILY, 10, "bold"), anchor="w",
                command=self._on_feature_toggle
            )
            cb.pack(fill="x", padx=8, pady=(6, 0))
            tk.Label(cell, text=FEATURE_DESCRIPTIONS[feat], bg=Theme.BG1,
                     fg=Theme.TEXT_MUTED, font=(Theme.FONT_FAMILY, 8),
                     wraplength=480, justify="left", anchor="w"
                     ).pack(fill="x", padx=32, pady=(0, 8))

        # ── Card: parámetros de entrenamiento ──
        param_card = make_card(scroll)
        param_card.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(param_card, text="Parámetros de entrenamiento", bg=Theme.BG2,
                 fg=Theme.TEXT_SECONDARY, font=(Theme.FONT_FAMILY, 9, "bold")
                 ).pack(anchor="w", padx=14, pady=(12, 6))

        pgrid = tk.Frame(param_card, bg=Theme.BG2)
        pgrid.pack(fill="x", padx=14, pady=(0, 6))
        pgrid.columnconfigure(0, weight=1)
        pgrid.columnconfigure(1, weight=1)

        self._form_slider(pgrid, 0, 0, "Profundidad máxima del árbol (max_depth)",
                          self.var_max_depth, 2, 10, lambda v: f"{int(float(v))}")
        self._form_slider(pgrid, 0, 1, "% de datos para prueba (test_size)",
                          self.var_test_size, 0.10, 0.40, lambda v: f"{float(v)*100:.0f}%")

        btn = tk.Label(param_card, text="⚙️  Aplicar y reentrenar modelo",
                       bg=Theme.ACCENT_BG, fg=Theme.ACCENT,
                       font=(Theme.FONT_FAMILY, 10, "bold"), pady=9, cursor="hand2")
        btn.pack(fill="x", padx=14, pady=(8, 14))
        btn.bind("<Button-1>", lambda e: self._aplicar_configuracion())

        # ── Nota pedagógica ──
        note = tk.Frame(scroll, bg=Theme.GOLD_BG, highlightbackground=Theme.GOLD,
                        highlightthickness=1)
        note.pack(fill="x", padx=16, pady=(0, 16))
        tk.Label(note, bg=Theme.GOLD_BG, fg=Theme.GOLD, justify="left", anchor="w",
                 font=(Theme.FONT_FAMILY, 8),
                 text="🎓  Conexión con el curso: al reducir variables (X) o aumentar la "
                      "profundidad del árbol puedes observar cómo cambia el desempeño en "
                      "Entrenamiento vs Prueba — visítalo en la pestaña correspondiente para "
                      "identificar señales de sobreajuste (overfitting) o subajuste (underfitting).",
                 wraplength=1100).pack(fill="x", padx=12, pady=10)
        return view

    def _on_feature_toggle(self):
        n_sel = sum(1 for f in ALL_FEATURES if self.feature_vars[f].get())
        if self.feature_count_label is not None:
            self.feature_count_label.configure(
                text=f"{n_sel} de {len(ALL_FEATURES)} variables seleccionadas"
            )

    # ── SEGMENTOS ─────────────────────────────────────────────────────────────
    def _build_segmentos_view(self):
        view   = tk.Frame(self.main_container, bg=Theme.BG0)
        scroll = self._scrollable(view)
        tk.Label(scroll, text="Análisis de segmentos", bg=Theme.BG0,
                 fg=Theme.TEXT_PRIMARY, font=(Theme.FONT_FAMILY, 12, "bold")
                 ).pack(anchor="w", padx=16, pady=(14, 10))
        grid = tk.Frame(scroll, bg=Theme.BG0)
        grid.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        segs = self._calcular_segmentos()
        for i, seg in enumerate(segs):
            r, c = divmod(i, 2)
            brd = Theme.SUCCESS if seg["badge_kind"] == "success" else Theme.BORDER
            card = make_card(grid, highlightbackground=brd,
                             highlightthickness=2 if seg["badge_kind"]=="success" else 1)
            card.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
            grid.rowconfigure(r, weight=1)
            top = tk.Frame(card, bg=Theme.BG2)
            top.pack(fill="x", padx=12, pady=(12, 0))
            left = tk.Frame(top, bg=Theme.BG2)
            left.pack(side="left", anchor="w")
            tk.Label(left, text=seg["nombre"], bg=Theme.BG2, fg=Theme.TEXT_PRIMARY,
                     font=(Theme.FONT_FAMILY, 10, "bold")).pack(anchor="w")
            tk.Label(left, text=seg["desc"], bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 8)).pack(anchor="w")
            make_badge(top, seg["badge_txt"], seg["badge_kind"]).pack(side="right")
            stats = tk.Frame(card, bg=Theme.BG2)
            stats.pack(fill="x", padx=12, pady=10)
            for j in range(2):
                stats.columnconfigure(j, weight=1)
            items = [
                ("Prob. conversión", f"{seg['prob']:.0f}%",          seg["prob_color"]),
                ("CTR promedio",     f"{seg['ctr']:.1f}%",           Theme.TEXT_PRIMARY),
                ("Tiempo sesión",    f"{seg['tiempo']:.1f} min",     Theme.TEXT_PRIMARY),
                ("Prospectos",       f"{seg['n']}",                   Theme.TEXT_PRIMARY),
            ]
            for k, (lbl, val, col) in enumerate(items):
                rr, cc = divmod(k, 2)
                cell = tk.Frame(stats, bg=Theme.BG2)
                cell.grid(row=rr, column=cc, sticky="w", pady=4)
                tk.Label(cell, text=lbl, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                         font=(Theme.FONT_FAMILY, 7)).pack(anchor="w")
                tk.Label(cell, text=val, bg=Theme.BG2, fg=col,
                         font=(Theme.FONT_FAMILY, 13, "bold")).pack(anchor="w")
        return view

    def _calcular_segmentos(self):
        df = self.model.df_bal.copy()
        df["disp_label"] = df["dispositivo"].map(DISPOSITIVO_MAP)
        combos  = [(2,"Escritorio"),(3,"Escritorio"),(1,"Móvil"),(1,"Tablet")]
        nombres = ["Segmento A","Segmento B","Segmento C","Segmento D"]
        res = []
        for (edad_cod, disp_label), nombre in zip(combos, nombres):
            sub = df[(df["edad_segmento"]==edad_cod) & (df["disp_label"]==disp_label)]
            if len(sub) == 0:
                continue
            prob = sub["convirtio"].mean() * 100
            if prob >= 55:   kind, txt, col = "success", "Alto potencial",    Theme.SUCCESS
            elif prob >= 35: kind, txt, col = "muted",   "Neutral",           Theme.TEXT_PRIMARY
            else:            kind, txt, col = "warning",  "Bajo rendimiento", Theme.WARNING
            res.append({"nombre": nombre,
                        "desc": f"{EDAD_MAP[edad_cod]} · {disp_label}",
                        "badge_kind": kind, "badge_txt": txt,
                        "prob": prob, "prob_color": col,
                        "ctr": sub["ctr"].mean(), "tiempo": sub["tiempo_sesion"].mean(),
                        "n": len(sub)})
        return res

    # ── PREDICCIÓN ────────────────────────────────────────────────────────────
    def _build_prediccion_view(self):
        view   = tk.Frame(self.main_container, bg=Theme.BG0)
        scroll = self._scrollable(view)
        tk.Label(scroll, text="Predecir conversión de un prospecto nuevo",
                 bg=Theme.BG0, fg=Theme.TEXT_PRIMARY,
                 font=(Theme.FONT_FAMILY, 12, "bold")
                 ).pack(anchor="w", padx=16, pady=(14, 10))

        form_card = make_card(scroll)
        form_card.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(form_card, text="Ingresar datos del prospecto (se usarán solo las variables activas en Configuración)",
                 bg=Theme.BG2, fg=Theme.TEXT_SECONDARY,
                 font=(Theme.FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=14, pady=(12, 10))

        grid = tk.Frame(form_card, bg=Theme.BG2)
        grid.pack(fill="x", padx=14, pady=(0, 6))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.var_clics       = tk.IntVar(value=5)
        self.var_ctr         = tk.DoubleVar(value=3.8)
        self.var_tiempo      = tk.DoubleVar(value=2.5)
        self.var_paginas     = tk.IntVar(value=3)
        self.var_dispositivo = tk.StringVar(value="Escritorio")
        self.var_edad        = tk.StringVar(value="25–34 años")

        self._form_spinbox(grid, 0, 0, "Clics en anuncio",    self.var_clics,   0,   19,  1)
        self._form_spinbox(grid, 0, 1, "CTR (%)",             self.var_ctr,     0.5, 7.0, 0.1)
        self._form_slider(grid, 1, 0, "Tiempo de sesión (min)", self.var_tiempo, 0.1, 15.0,
                          lambda v: f"{float(v):.1f} min")
        self._form_slider(grid, 1, 1, "Páginas vistas",       self.var_paginas, 1,   11,
                          lambda v: f"{int(float(v))}")
        self._form_combo(grid, 2, 0, "Dispositivo",           self.var_dispositivo,
                         ["Escritorio","Móvil","Tablet"])
        self._form_combo(grid, 2, 1, "Segmento etario",       self.var_edad,
                         ["25–34 años","18–24 años","35–44 años","45+ años"])

        btn = tk.Label(form_card, text="🧠  Clasificar con modelo ML",
                       bg=Theme.ACCENT_BG, fg=Theme.ACCENT,
                       font=(Theme.FONT_FAMILY, 10, "bold"), pady=9, cursor="hand2")
        btn.pack(fill="x", padx=14, pady=(10, 14))
        btn.bind("<Button-1>", lambda e: self._ejecutar_prediccion())

        self.pred_result_card  = make_card(scroll)
        self.pred_result_inner = None
        return view

    def _form_spinbox(self, parent, r, c, label, var, lo, hi, step):
        cell = tk.Frame(parent, bg=Theme.BG2)
        cell.grid(row=r, column=c, sticky="ew", padx=8, pady=6)
        tk.Label(cell, text=label, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                 font=(Theme.FONT_FAMILY, 8)).pack(anchor="w")
        tk.Spinbox(cell, from_=lo, to=hi, increment=step, textvariable=var,
                   bg=Theme.BG1, fg=Theme.TEXT_PRIMARY,
                   insertbackground=Theme.TEXT_PRIMARY,
                   highlightbackground=Theme.BORDER, highlightthickness=1,
                   relief="flat", buttonbackground=Theme.BG1,
                   font=(Theme.FONT_FAMILY, 10)).pack(fill="x", pady=(2, 0))

    def _form_slider(self, parent, r, c, label, var, lo, hi, fmt_fn):
        cell = tk.Frame(parent, bg=Theme.BG2)
        cell.grid(row=r, column=c, sticky="ew", padx=8, pady=6)
        top = tk.Frame(cell, bg=Theme.BG2)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                 font=(Theme.FONT_FAMILY, 8)).pack(side="left")
        out = tk.Label(top, text=fmt_fn(var.get()), bg=Theme.BG2,
                       fg=Theme.TEXT_SECONDARY, font=(Theme.FONT_FAMILY, 8))
        out.pack(side="right")
        resolution = 0.01 if isinstance(var, tk.DoubleVar) and hi <= 1 else (
            0.1 if isinstance(var, tk.DoubleVar) else 1)
        tk.Scale(cell, from_=lo, to=hi, orient="horizontal", variable=var,
                 showvalue=False, bg=Theme.BG2, fg=Theme.TEXT_PRIMARY,
                 troughcolor=Theme.BG1, highlightthickness=0, bd=0,
                 activebackground=Theme.ACCENT, sliderrelief="flat",
                 resolution=resolution,
                 command=lambda v: out.configure(text=fmt_fn(v))
                 ).pack(fill="x", pady=(2, 0))

    def _form_combo(self, parent, r, c, label, var, options):
        cell = tk.Frame(parent, bg=Theme.BG2)
        cell.grid(row=r, column=c, sticky="ew", padx=8, pady=6)
        tk.Label(cell, text=label, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                 font=(Theme.FONT_FAMILY, 8)).pack(anchor="w")
        ttk.Combobox(cell, textvariable=var, values=options, state="readonly",
                     font=(Theme.FONT_FAMILY, 10)).pack(fill="x", pady=(2, 0))

    def _ejecutar_prediccion(self):
        disp_inv = {v: k for k, v in DISPOSITIVO_MAP.items()}
        edad_inv = {v: k for k, v in EDAD_MAP.items()}
        try:
            clics  = int(self.var_clics.get())
            ctr    = float(self.var_ctr.get())
            tiempo = float(self.var_tiempo.get())
            pags   = int(self.var_paginas.get())
            disp   = disp_inv[self.var_dispositivo.get()]
            edad   = edad_inv[self.var_edad.get()]
        except Exception as ex:
            messagebox.showerror("Error", f"Valor inválido en el formulario:\n{ex}")
            return

        pred, proba = self.model.predecir(clics, ctr, tiempo, pags, disp, edad)
        prob_conv   = proba[1] * 100

        if prob_conv >= 65:
            label, color, icon = "Convertirá",          Theme.SUCCESS, "✓"
            rec = "Priorizar este prospecto. Asignar hasta el 30 % del presupuesto del día."
        elif prob_conv >= 40:
            label, color, icon = "Probable conversión", Theme.WARNING, "!"
            rec = "Incluir en remarketing con oferta específica para aumentar la probabilidad."
        else:
            label, color, icon = "No convertirá",       Theme.DANGER,  "✗"
            rec = "Excluir del presupuesto principal. Considerar campaña de awareness de bajo costo."

        if self.pred_result_inner:
            self.pred_result_inner.destroy()
        self.pred_result_card.pack(fill="x", padx=16, pady=(0, 16))
        inner = tk.Frame(self.pred_result_card, bg=Theme.BG2)
        inner.pack(fill="x", padx=14, pady=12)
        self.pred_result_inner = inner

        tk.Label(inner, text="Resultado de la clasificación", bg=Theme.BG2,
                 fg=Theme.TEXT_SECONDARY, font=(Theme.FONT_FAMILY, 9, "bold")
                 ).pack(anchor="w", pady=(0, 10))
        top = tk.Frame(inner, bg=Theme.BG2)
        top.pack(fill="x")
        tk.Label(top, text=icon, bg=Theme.BG2, fg=color,
                 font=(Theme.FONT_FAMILY, 22, "bold")).pack(side="left", padx=(0, 12))
        txt = tk.Frame(top, bg=Theme.BG2)
        txt.pack(side="left", anchor="w")
        tk.Label(txt, text=label, bg=Theme.BG2, fg=color,
                 font=(Theme.FONT_FAMILY, 14, "bold")).pack(anchor="w")
        tk.Label(txt, text=f"Probabilidad estimada: {prob_conv:.1f}%  ·  variables usadas: "
                           f"{', '.join(FEATURE_LABELS[f] for f in self.model.features_used)}",
                 bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                 font=(Theme.FONT_FAMILY, 9), wraplength=1000, justify="left").pack(anchor="w")
        rec_box = tk.Frame(inner, bg=Theme.BG1)
        rec_box.pack(fill="x", pady=(12, 0))
        tk.Label(rec_box, text="💡  Recomendación de acción", bg=Theme.BG1,
                 fg=Theme.TEXT_SECONDARY, font=(Theme.FONT_FAMILY, 9, "bold")
                 ).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(rec_box, text=rec, bg=Theme.BG1, fg=Theme.TEXT_PRIMARY,
                 font=(Theme.FONT_FAMILY, 9), wraplength=1000, justify="left"
                 ).pack(anchor="w", padx=10, pady=(0, 8))
        self.status_text.configure(text=f"Predicción: {label} ({prob_conv:.1f}%)")

    # ── ENTRENAMIENTO VS PRUEBA ────────────────────────────────────────────────
    def _build_comparacion_view(self):
        view   = tk.Frame(self.main_container, bg=Theme.BG0)
        scroll = self._scrollable(view)
        m      = self.model
        tr, te = m.train_metrics, m.test_metrics

        header = tk.Frame(scroll, bg=Theme.BG0)
        header.pack(fill="x", padx=16, pady=(14, 4))
        tk.Label(header, text="Entrenamiento vs Prueba", bg=Theme.BG0,
                 fg=Theme.TEXT_PRIMARY, font=(Theme.FONT_FAMILY, 12, "bold")).pack(side="left")
        make_badge(header, f"  {tr['n']} train · {te['n']} test  ", "accent").pack(side="right")

        tk.Label(scroll,
                 text="Comparar el desempeño del modelo en los datos con los que aprendió "
                      "(entrenamiento) frente a datos que no vio (prueba) permite detectar "
                      "sobreajuste: si el modelo memoriza en vez de generalizar.",
                 bg=Theme.BG0, fg=Theme.TEXT_MUTED, font=(Theme.FONT_FAMILY, 8),
                 wraplength=1100, justify="left").pack(anchor="w", padx=16, pady=(0, 12))

        # KPIs comparativos
        kpi_wrap = tk.Frame(scroll, bg=Theme.BG0)
        kpi_wrap.pack(fill="x", padx=16, pady=(0, 10))
        for i in range(4):
            kpi_wrap.columnconfigure(i, weight=1)
        metrics_labels = [("accuracy","Accuracy"),("precision","Precision"),
                          ("recall","Recall"),("f1","F1 Score")]
        for i, (key, lbl) in enumerate(metrics_labels):
            card = make_card(kpi_wrap)
            card.grid(row=0, column=i, sticky="nsew", padx=5)
            tk.Label(card, text=lbl, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 8)).pack(anchor="w", padx=12, pady=(10, 4))
            v_tr, v_te = tr[key]*100, te[key]*100
            row = tk.Frame(card, bg=Theme.BG2)
            row.pack(anchor="w", padx=12, pady=(0, 4))
            tk.Label(row, text=f"{v_tr:.1f}%", bg=Theme.BG2, fg=Theme.BLUE,
                     font=(Theme.FONT_FAMILY, 15, "bold")).pack(side="left")
            tk.Label(row, text=" train  ", bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 8)).pack(side="left")
            tk.Label(row, text=f"{v_te:.1f}%", bg=Theme.BG2, fg=Theme.ACCENT,
                     font=(Theme.FONT_FAMILY, 15, "bold")).pack(side="left")
            tk.Label(row, text=" test", bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 8)).pack(side="left")
            gap = v_tr - v_te
            gap_color = Theme.DANGER if gap > 12 else (Theme.WARNING if gap > 6 else Theme.SUCCESS)
            tk.Label(card, text=f"brecha: {gap:+.1f} pts", bg=Theme.BG2, fg=gap_color,
                     font=(Theme.FONT_FAMILY, 8, "bold")).pack(anchor="w", padx=12, pady=(0, 10))

        # Gráfico comparativo de barras agrupadas
        chart_grid = tk.Frame(scroll, bg=Theme.BG0)
        chart_grid.pack(fill="x", padx=16, pady=(0, 10))
        chart_grid.columnconfigure(0, weight=1)
        chart_grid.columnconfigure(1, weight=1)

        bar_card = make_card(chart_grid)
        bar_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(bar_card, text="📊  Comparación de métricas", bg=Theme.BG2,
                 fg=Theme.TEXT_SECONDARY, font=(Theme.FONT_FAMILY, 9, "bold")
                 ).pack(anchor="w", padx=12, pady=(10, 4))
        self._embed_comparison_chart(bar_card, tr, te)

        cm_card = make_card(chart_grid)
        cm_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(cm_card, text="🔢  Matrices de confusión", bg=Theme.BG2,
                 fg=Theme.TEXT_SECONDARY, font=(Theme.FONT_FAMILY, 9, "bold")
                 ).pack(anchor="w", padx=12, pady=(10, 8))
        cm_row = tk.Frame(cm_card, bg=Theme.BG2)
        cm_row.pack(fill="x", padx=12, pady=(0, 12))
        cm_row.columnconfigure(0, weight=1)
        cm_row.columnconfigure(1, weight=1)
        self._confusion_matrix_widget(cm_row, tr["confusion_matrix"], "Entrenamiento", Theme.BLUE, 0)
        self._confusion_matrix_widget(cm_row, te["confusion_matrix"], "Prueba", Theme.ACCENT, 1)

        # Lectura pedagógica automática
        acc_gap = (tr["accuracy"] - te["accuracy"]) * 100
        if acc_gap > 12:
            veredicto = ("Posible sobreajuste (overfitting): el modelo memoriza patrones del "
                         "conjunto de entrenamiento y generaliza peor en datos nuevos. Prueba "
                         "reducir la profundidad del árbol o el número de variables.")
            v_kind, v_color = Theme.DANGER_BG, Theme.DANGER
        elif acc_gap > 6:
            veredicto = ("Brecha moderada entre entrenamiento y prueba. Vigila el desempeño al "
                         "cambiar variables o profundidad del árbol.")
            v_kind, v_color = Theme.WARNING_BG, Theme.WARNING
        elif te["accuracy"] < 0.6:
            veredicto = ("El modelo generaliza de forma pareja, pero la exactitud general es "
                         "baja: posible subajuste (underfitting). Considera agregar variables o "
                         "aumentar la profundidad del árbol.")
            v_kind, v_color = Theme.WARNING_BG, Theme.WARNING
        else:
            veredicto = ("Buen balance entre entrenamiento y prueba: el modelo generaliza "
                         "adecuadamente con las variables y parámetros actuales.")
            v_kind, v_color = Theme.SUCCESS_BG, Theme.SUCCESS

        concl = tk.Frame(scroll, bg=v_kind, highlightbackground=v_color, highlightthickness=1)
        concl.pack(fill="x", padx=16, pady=(0, 16))
        tk.Label(concl, bg=v_kind, fg=v_color, justify="left", anchor="w",
                 font=(Theme.FONT_FAMILY, 9, "bold"),
                 text=f"🎓  Lectura del curso: {veredicto}",
                 wraplength=1100).pack(fill="x", padx=12, pady=10)
        return view

    def _embed_comparison_chart(self, parent, tr, te):
        fig = Figure(figsize=(4.6, 2.8), dpi=100)
        fig.patch.set_facecolor(Theme.BG2)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(Theme.BG2)
        labels = ["Accuracy", "Precision", "Recall", "F1"]
        keys   = ["accuracy", "precision", "recall", "f1"]
        vals_tr = [tr[k] for k in keys]
        vals_te = [te[k] for k in keys]
        x = np.arange(len(labels))
        w = 0.35
        b1 = ax.bar(x - w/2, vals_tr, width=w, color=Theme.BLUE, label="Entrenamiento")
        b2 = ax.bar(x + w/2, vals_te, width=w, color=Theme.ACCENT, label="Prueba")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, color=Theme.TEXT_MUTED)
        ax.set_ylim(0, 1.15)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h+0.03, f"{h*100:.0f}%",
                        ha="center", va="bottom", fontsize=7, color=Theme.TEXT_PRIMARY)
        ax.tick_params(colors=Theme.TEXT_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
        legend = ax.legend(fontsize=7, facecolor=Theme.BG2, edgecolor=Theme.BORDER, loc="lower right")
        for text in legend.get_texts():
            text.set_color(Theme.TEXT_SECONDARY)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=parent).get_tk_widget().pack(
            fill="both", expand=True, padx=8, pady=(0, 8))

    def _confusion_matrix_widget(self, parent, cm, title, color, col):
        wrap = tk.Frame(parent, bg=Theme.BG2)
        wrap.grid(row=0, column=col, sticky="n", padx=6)
        tk.Label(wrap, text=title, bg=Theme.BG2, fg=color,
                 font=(Theme.FONT_FAMILY, 9, "bold")).pack(pady=(0, 6))
        grid = tk.Frame(wrap, bg=Theme.BG2)
        grid.pack()
        etiquetas = ["No convirtió", "Convirtió"]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        celdas = [[tn, fp], [fn, tp]]
        tk.Label(grid, text="", bg=Theme.BG2, width=10).grid(row=0, column=0)
        for j, et in enumerate(etiquetas):
            tk.Label(grid, text=et, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 7), width=10).grid(row=0, column=j+1)
        for i, et in enumerate(etiquetas):
            tk.Label(grid, text=et, bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 7), width=10).grid(row=i+1, column=0)
            for j in range(2):
                val = celdas[i][j]
                bg = Theme.SUCCESS_BG if i == j else Theme.DANGER_BG
                fg = Theme.SUCCESS if i == j else Theme.DANGER
                tk.Label(grid, text=str(val), bg=bg, fg=fg, width=10, height=2,
                         font=(Theme.FONT_FAMILY, 10, "bold"),
                         highlightbackground=Theme.BORDER, highlightthickness=1
                         ).grid(row=i+1, column=j+1, padx=2, pady=2)

    # ── REGLAS DEL ÁRBOL ────────────────────────────────────────────────────────
    def _build_reglas_view(self):
        view   = tk.Frame(self.main_container, bg=Theme.BG0)
        scroll = self._scrollable(view)
        m      = self.model

        header = tk.Frame(scroll, bg=Theme.BG0)
        header.pack(fill="x", padx=16, pady=(14, 10))
        tk.Label(header, text="Reglas del árbol de decisión", bg=Theme.BG0,
                 fg=Theme.TEXT_PRIMARY, font=(Theme.FONT_FAMILY, 12, "bold")).pack(side="left")
        make_badge(header, f"  max_depth={m.clf.get_depth()} · Gini · {len(m.df_bal)} registros  ",
                   "accent").pack(side="right")

        info = tk.Frame(scroll, bg=Theme.BG1)
        info.pack(fill="x", padx=16, pady=(0, 10))
        variables_txt = ", ".join(FEATURE_LABELS[f] for f in m.features_used)
        tk.Label(info, bg=Theme.BG1, fg=Theme.TEXT_SECONDARY, justify="left", anchor="w",
                 font=(Theme.FONT_FAMILY, 8),
                 text=f"ℹ  Árbol entrenado con las variables: {variables_txt}. "
                      "El equipo puede revisar estas reglas para entender cómo SmartSeg Ads "
                      "clasifica cada prospecto sin necesidad de conocimientos avanzados de "
                      "matemáticas.",
                 wraplength=1080).pack(fill="x", padx=10, pady=8)

        rules_card = make_card(scroll)
        rules_card.pack(fill="x", padx=16, pady=(0, 12))
        tk.Label(rules_card, text="Árbol completo (export_text — scikit-learn)",
                 bg=Theme.BG2, fg=Theme.TEXT_SECONDARY,
                 font=(Theme.FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        tf = tk.Frame(rules_card, bg=Theme.BG2)
        tf.pack(fill="x", padx=12, pady=(0, 12))
        rt = tk.Text(tf, height=14, bg=Theme.BG1, fg=Theme.TEXT_SECONDARY,
                     font=(Theme.FONT_MONO, 9), wrap="none", relief="flat",
                     highlightbackground=Theme.BORDER, highlightthickness=1,
                     padx=10, pady=8)
        vsb = ttk.Scrollbar(tf, orient="vertical",   command=rt.yview)
        hsb = ttk.Scrollbar(tf, orient="horizontal",  command=rt.xview)
        rt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        rt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tf.rowconfigure(0, weight=1)
        tf.columnconfigure(0, weight=1)
        rt.insert("1.0", m.rules_text)
        rt.configure(state="disabled")

        imp_card = make_card(scroll)
        imp_card.pack(fill="x", padx=16, pady=(0, 16))
        tk.Label(imp_card, text="Importancia de variables (Feature Importance — Gini real)",
                 bg=Theme.BG2, fg=Theme.TEXT_SECONDARY,
                 font=(Theme.FONT_FAMILY, 9, "bold")).pack(anchor="w", padx=12, pady=(10, 8))
        imp_inner = tk.Frame(imp_card, bg=Theme.BG2)
        imp_inner.pack(fill="x", padx=12, pady=(0, 6))
        cols_cycle = [Theme.BLUE, Theme.GREEN, Theme.PURPLE, Theme.ORANGE,
                      Theme.BORDER_STRONG, Theme.BORDER_STRONG]
        max_imp = m.importances.max() if len(m.importances) else 0
        for i, (feat, imp) in enumerate(m.importances.items()):
            pct = (imp/max_imp)*100 if max_imp > 0 else 0
            row = tk.Frame(imp_inner, bg=Theme.BG2)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=FEATURE_LABELS.get(feat, feat), bg=Theme.BG2,
                     fg=Theme.TEXT_PRIMARY, font=(Theme.FONT_FAMILY, 9),
                     width=18, anchor="w").pack(side="left")
            bar = tk.Canvas(row, height=8, bg=Theme.BG1, highlightthickness=0)
            bar.pack(side="left", padx=8, fill="x", expand=True)

            def make_drawer(canvas_ref, p, col):
                def draw(event=None):
                    canvas_ref.delete("all")
                    w  = canvas_ref.winfo_width() or 300
                    fw = max(2, int(w * min(max(p,0),100)/100))
                    canvas_ref.create_rectangle(0,0,w,8,  fill=Theme.BG1, outline="")
                    canvas_ref.create_rectangle(0,0,fw,8, fill=col,       outline="")
                return draw

            bar.bind("<Configure>", make_drawer(bar, pct, cols_cycle[i%len(cols_cycle)]))
            tk.Label(row, text=f"{imp:.3f}", bg=Theme.BG2, fg=Theme.TEXT_MUTED,
                     font=(Theme.FONT_FAMILY, 9), width=6, anchor="e").pack(side="left")

        btn_lbl = ("🔄  Generar nueva muestra simulada y reentrenar"
                   if m.fuente == "simulado"
                   else "🔄  Reentrenar con los mismos datos Excel cargados")
        btn = tk.Label(imp_card, text=btn_lbl, bg=Theme.BG1, fg=Theme.TEXT_PRIMARY,
                       font=(Theme.FONT_FAMILY, 9, "bold"), pady=8, cursor="hand2")
        btn.pack(fill="x", padx=12, pady=(10, 12))
        btn.bind("<Button-1>", lambda e: self._reentrenar_simulado())
        return view

    # ── SCROLL HELPER ─────────────────────────────────────────────────────────
    def _scrollable(self, parent):
        canvas  = tk.Canvas(parent, bg=Theme.BG0, highlightthickness=0)
        vsb     = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner   = tk.Frame(canvas, bg=Theme.BG0)
        win_id  = canvas.create_window((0,0), window=inner, anchor="nw")

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.bind("<Enter>",
                    lambda e: canvas.bind_all("<MouseWheel>",
                        lambda ev: canvas.yview_scroll(int(-1*(ev.delta/120)), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner


# ==============================================================================
#  PUNTO DE ENTRADA
# ==============================================================================
def main():
    SmartSegApp().mainloop()


if __name__ == "__main__":
    main()
