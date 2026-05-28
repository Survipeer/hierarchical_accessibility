"""
accessibility_summary.py
Counts reachable (origin, dest) hex pairs per travel-time bucket across
a chosen time window, then displays counts + percentages in a dark-themed GUI.

Input (via GUI)
---------------
  Result folder   – one JSON per origin hex, produced by
                    reachability_hierarchical.py or
                    reachability_analysis_hex2hex.py
  Hex centroid CSV – any CSV that contains H3 hex IDs
  Start / End time – HH:MM window; only departure intervals in this
                    range are counted

JSON format expected
--------------------
  {
    "origin_hex": "...",
    "walk_results":   { "destination": [{"hex_id": ...}, ...] },
    "results_by_time": {
      "YYYY-MM-DD HH:MM:SS": {
        "0-15 min":  { "destination": [{"hex_id": ...}, ...] },
        "15-30 min": { ... },
        "30-45 min": { ... },
        "45-60 min": { ... }
      }, ...
    }
  }

Counting logic
--------------
  For each origin hex:
    For each qualifying departure interval:
      For each bucket:
        Add every (origin, dest_hex_id) pair to that bucket's set.
  Result = size of each set (each unique pair counted once no matter
  how many departure times it appears in).
  Walk-only = walk_results pairs not already in any transit bucket.
  Unreachable = total_possible - union(all_buckets | walk).
"""

import json, os, csv, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, time as dtime

# ── constants ─────────────────────────────────────────────────────────────────
BUCKETS       = ["0-15 min", "15-30 min", "30-45 min", "45-60 min"]
BUCKET_LABELS = ["0–15 min",  "15–30 min", "30–45 min", "45–60 min"]
B_COLORS      = ["#22c55e",   "#38bdf8",   "#818cf8",   "#f59e0b"]

# dark palette
C = dict(
    BG="#0d1117", PANEL="#161b22", CARD="#1c2333",
    ACCENT="#58a6ff", TEXT="#c9d1d9", MUTED="#484f58",
    ENTRY="#0d1117", GREEN="#3fb950", RED="#f85149",
    AMBER="#d29922", BORDER="#30363d"
)


# ── analysis helpers ──────────────────────────────────────────────────────────

def parse_hhmm(s):
    try:
        h, m = s.strip().split(":")
        return dtime(int(h), int(m))
    except Exception:
        raise ValueError(f"Cannot parse time '{s}' — expected HH:MM")


def detect_hex_column(headers, first_row):
    """Return the column name that holds H3 hex IDs."""
    for name in ["hex_id", "h3_id", "hexid", "hex", "id"]:
        match = next((h for h in headers if h.strip().lower() == name), None)
        if match:
            return match
    # auto-detect: 15-char string starting with '8'
    for col in headers:
        v = first_row.get(col, "").strip()
        if len(v) == 15 and v.startswith("8"):
            return col
    return headers[0]  # fallback


def load_hex_ids(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return set()
    col = detect_hex_column(headers, rows[0])
    return {r[col].strip() for r in rows if r.get(col, "").strip()}


def run_analysis(folder, csv_path, start_t, end_t, progress_cb=None):
    """
    Returns dict:
      total_hexes, total_possible,
      buckets {label: count}, walk_only, transit_any, unreachable,
      files_processed, files_skipped, n_files
    """
    hex_ids        = load_hex_ids(csv_path)
    total_hexes    = len(hex_ids)
    total_possible = total_hexes * (total_hexes - 1)

    bucket_pairs = {b: set() for b in BUCKETS}
    walk_pairs   = set()

    json_files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    n_files    = len(json_files)
    processed  = skipped = 0

    for fi, fname in enumerate(json_files):
        if progress_cb:
            progress_cb(fi, n_files)

        try:
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception:
            skipped += 1
            continue

        # --- FIX 1: Handle if the entire JSON is wrapped in a list ---
        if isinstance(raw_data, list):
            records = raw_data
        elif isinstance(raw_data, dict):
            records = [raw_data]
        else:
            skipped += 1
            continue

        file_has_valid_data = False

        for data in records:
            # Extra safety: ensure the record is a dictionary before proceeding
            if not isinstance(data, dict):
                continue

            origin = data.get("origin_hex", "")
            if not origin or origin not in hex_ids:
                continue

            # --- FIX 2: Bulletproof Walk extraction ---
            walk_data = data.get("walk_results", {})
            walk_list = []
            
            if isinstance(walk_data, dict):
                walk_list = walk_data.get("destination", [])
            elif isinstance(walk_data, list):
                walk_list = walk_data
            
            if not isinstance(walk_list, list):
                walk_list = []

            for entry in walk_list:
                dest = entry.get("hex_id", "") if isinstance(entry, dict) else str(entry)
                if dest and dest in hex_ids and dest != origin:
                    walk_pairs.add((origin, dest))

            # --- FIX 3: Bulletproof Transit extraction ---
            results_time = data.get("results_by_time", {})
            
            if isinstance(results_time, dict):
                for interval_str, buckets in results_time.items():
                    if not isinstance(buckets, dict):
                        continue
                        
                    try:
                        itime = datetime.strptime(str(interval_str), "%Y-%m-%d %H:%M:%S").time()
                    except Exception:
                        try:
                            itime = dtime.fromisoformat(str(interval_str)[:5])
                        except Exception:
                            continue

                    if not (start_t <= itime <= end_t):
                        continue

                    for bucket in BUCKETS:
                        b_data = buckets.get(bucket, {})
                        b_list = []
                        
                        if isinstance(b_data, dict):
                            b_list = b_data.get("destination", [])
                        elif isinstance(b_data, list):
                            b_list = b_data
                            
                        if not isinstance(b_list, list):
                            b_list = []

                        for entry in b_list:
                            dest = entry.get("hex_id", "") if isinstance(entry, dict) else str(entry)
                            if dest and dest in hex_ids and dest != origin:
                                bucket_pairs[bucket].add((origin, dest))
            
            file_has_valid_data = True

        if file_has_valid_data:
            processed += 1
        else:
            skipped += 1

    if progress_cb:
        progress_cb(n_files, n_files)

    transit_reachable = set().union(*bucket_pairs.values())
    all_reachable     = transit_reachable | walk_pairs
    unreachable       = max(0, total_possible - len(all_reachable))

    return dict(
        total_hexes     = total_hexes,
        total_possible  = total_possible,
        buckets         = {b: len(bucket_pairs[b]) for b in BUCKETS},
        walk_only       = len(walk_pairs - transit_reachable),
        transit_any     = len(transit_reachable),
        unreachable     = unreachable,
        files_processed = processed,
        files_skipped   = skipped,
        n_files         = n_files,
    )

# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transit Accessibility Summary")
        self.configure(bg=C["BG"])
        self.geometry("860x700")
        self.minsize(700, 560)
        self._results = None
        self._build()

    # ─── construction ─────────────────────────────────────────────────────────

    def _build(self):
        # title strip
        strip = tk.Frame(self, bg=C["PANEL"], pady=14)
        strip.pack(fill="x")
        tk.Label(strip, text="Transit Accessibility Summary",
                 font=("Segoe UI", 15, "bold"),
                 bg=C["PANEL"], fg=C["ACCENT"]).pack()
        tk.Label(strip,
                 text="Count unique reachable hex pairs in each travel-time bucket",
                 font=("Segoe UI", 9),
                 bg=C["PANEL"], fg=C["MUTED"]).pack()

        # inputs
        inp = tk.Frame(self, bg=C["BG"], padx=28, pady=16)
        inp.pack(fill="x")
        inp.columnconfigure(1, weight=1)

        self.v_folder = tk.StringVar()
        self.v_csv    = tk.StringVar()
        self.v_start  = tk.StringVar(value="00:00")
        self.v_end    = tk.StringVar(value="23:45")

        def field(r, label, var, browse_fn=None, hint=""):
            tk.Label(inp, text=label, anchor="w", width=22,
                     font=("Segoe UI", 10, "bold"),
                     bg=C["BG"], fg=C["TEXT"]
                     ).grid(row=r, column=0, sticky="w", pady=5)
            e = tk.Entry(inp, textvariable=var,
                         bg=C["ENTRY"], fg=C["TEXT"],
                         insertbackground=C["TEXT"],
                         highlightthickness=1,
                         highlightcolor=C["ACCENT"],
                         highlightbackground=C["BORDER"],
                         relief="flat",
                         font=("Segoe UI", 10))
            e.grid(row=r, column=1, sticky="ew", padx=(6, 4))
            if browse_fn:
                tk.Button(inp, text="Browse",
                          command=browse_fn,
                          bg=C["ACCENT"], fg=C["BG"],
                          relief="flat", cursor="hand2",
                          font=("Segoe UI", 9, "bold"),
                          padx=10, pady=2
                          ).grid(row=r, column=2, padx=(0, 0))
            if hint:
                tk.Label(inp, text=hint,
                         font=("Segoe UI", 8),
                         bg=C["BG"], fg=C["MUTED"]
                         ).grid(row=r, column=3, padx=10, sticky="w")

        field(0, "Results Folder",   self.v_folder,
              lambda: self._browse_dir(self.v_folder))
        field(1, "Hex Centroid CSV", self.v_csv,
              lambda: self._browse_file(self.v_csv))
        field(2, "Start Time",       self.v_start, hint="HH:MM")
        field(3, "End Time",         self.v_end,   hint="HH:MM")

        # buttons
        bf = tk.Frame(self, bg=C["BG"], pady=8)
        bf.pack()
        self.btn_run = tk.Button(
            bf, text="▶  Run Analysis", command=self._run,
            bg=C["ACCENT"], fg=C["BG"], relief="flat",
            font=("Segoe UI", 11, "bold"),
            padx=28, pady=8, cursor="hand2")
        self.btn_run.pack(side="left", padx=6)
        self.btn_export = tk.Button(
            bf, text="⬇  Export CSV", command=self._export,
            bg=C["GREEN"], fg=C["BG"], relief="flat",
            font=("Segoe UI", 11, "bold"),
            padx=28, pady=8, cursor="hand2",
            state="disabled")
        self.btn_export.pack(side="left", padx=6)

        # progress
        pf = tk.Frame(self, bg=C["BG"], padx=28)
        pf.pack(fill="x")
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except:
            pass

        # Use existing built-in layout
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor=C["PANEL"],
            background=C["ACCENT"],
            thickness=6,
            bordercolor=C["PANEL"],
            lightcolor=C["ACCENT"],
            darkcolor=C["ACCENT"]
        )

        self.prog = ttk.Progressbar(
            pf,
            style="Custom.Horizontal.TProgressbar",
            mode="determinate"
        )
        self.prog.pack(fill="x", pady=(2, 1))
        self.lbl_prog = tk.Label(pf, text="",
                                 font=("Segoe UI", 8),
                                 bg=C["BG"], fg=C["MUTED"])
        self.lbl_prog.pack(anchor="w")

        # results pane
        rp = tk.Frame(self, bg=C["BG"], padx=28, pady=6)
        rp.pack(fill="both", expand=True)
        tk.Label(rp, text="OUTPUT",
                 font=("Segoe UI", 8, "bold"),
                 bg=C["BG"], fg=C["MUTED"]).pack(anchor="w")
        self.res_outer = tk.Frame(rp, bg=C["PANEL"],
                                  highlightthickness=1,
                                  highlightbackground=C["BORDER"])
        self.res_outer.pack(fill="both", expand=True, pady=4)
        self._placeholder()

    def _placeholder(self):
        tk.Label(self.res_outer,
                 text="Configure inputs above and click Run Analysis.",
                 font=("Segoe UI", 10),
                 bg=C["PANEL"], fg=C["MUTED"]).pack(expand=True, pady=40)

    # ─── event handlers ───────────────────────────────────────────────────────

    def _browse_dir(self, var):
        p = filedialog.askdirectory(title="Select results folder")
        if p:
            var.set(p)

    def _browse_file(self, var):
        p = filedialog.askopenfilename(
            title="Select hex centroid CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            var.set(p)

    def _run(self):
        folder = self.v_folder.get().strip()
        csv_p  = self.v_csv.get().strip()
        s_str  = self.v_start.get().strip()
        e_str  = self.v_end.get().strip()

        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Input error",
                                 "Please select a valid results folder.")
            return
        if not csv_p or not os.path.isfile(csv_p):
            messagebox.showerror("Input error",
                                 "Please select a valid hex centroid CSV.")
            return
        try:
            start_t = parse_hhmm(s_str)
            end_t   = parse_hhmm(e_str)
        except ValueError as ex:
            messagebox.showerror("Input error", str(ex))
            return
        if start_t > end_t:
            messagebox.showerror("Input error",
                                 "Start time must be ≤ end time.")
            return

        self.btn_run.config(state="disabled", text="Running…")
        self.btn_export.config(state="disabled")
        self.prog["value"] = 0
        self.lbl_prog.config(text="Starting…")
        self._clear_results()
        self._results = None

        def worker():
            def cb(done, total):
                pct = int(done / total * 100) if total else 0
                self.after(0, lambda d=done, t=total, p=pct:
                           self._prog_update(d, t, p))
            try:
                res = run_analysis(folder, csv_p, start_t, end_t, cb)
                self.after(0, lambda r=res: self._show_results(r))
            except Exception as ex:
                self.after(0, lambda e=str(ex): self._on_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _prog_update(self, done, total, pct):
        self.prog["value"] = pct
        self.lbl_prog.config(text=f"File {done} / {total}   ({pct}%)")

    def _on_error(self, msg):
        self.btn_run.config(state="normal", text="▶  Run Analysis")
        messagebox.showerror("Analysis failed", msg)

    # ─── results rendering ────────────────────────────────────────────────────

    def _clear_results(self):
        for w in self.res_outer.winfo_children():
            w.destroy()

    def _show_results(self, res):
        self._results = res
        self._clear_results()

        total_p = res["total_possible"]

        def pct(n):
            return n / total_p * 100 if total_p else 0.0

        def pct_str(n):
            return f"{pct(n):.2f}%"

        pad = dict(padx=16, pady=10)

        # ── stat cards row ─────────────────────────────────────────────────
        cards = tk.Frame(self.res_outer, bg=C["PANEL"])
        cards.pack(fill="x", **pad)

        stats = [
            ("Total Hexes",      f"{res['total_hexes']:,}",    C["TEXT"]),
            ("Possible Pairs",   f"{total_p:,}",               C["TEXT"]),
            ("Transit Reachable",f"{res['transit_any']:,}",    C["GREEN"],
             pct_str(res["transit_any"])),
            ("Walk Only",        f"{res['walk_only']:,}",      C["ACCENT"],
             pct_str(res["walk_only"])),
            ("Unreachable",      f"{res['unreachable']:,}",    C["RED"],
             pct_str(res["unreachable"])),
        ]
        for item in stats:
            label, value, color = item[0], item[1], item[2]
            sub = item[3] if len(item) > 3 else ""
            f = tk.Frame(cards, bg=C["CARD"],
                         highlightthickness=1,
                         highlightbackground=C["BORDER"],
                         padx=10, pady=8)
            f.pack(side="left", fill="both", expand=True, padx=3)
            tk.Label(f, text=value,
                     font=("Segoe UI", 15, "bold"),
                     bg=C["CARD"], fg=color).pack()
            tk.Label(f, text=label,
                     font=("Segoe UI", 8),
                     bg=C["CARD"], fg=C["MUTED"]).pack()
            if sub:
                tk.Label(f, text=sub,
                         font=("Segoe UI", 8, "italic"),
                         bg=C["CARD"], fg=color).pack()

        # ── file info ──────────────────────────────────────────────────────
        tk.Label(self.res_outer,
                 text=(f"  {res['files_processed']} files processed  ·  "
                       f"{res['files_skipped']} skipped  ·  "
                       f"{res['n_files']} total in folder"),
                 font=("Segoe UI", 8),
                 bg=C["PANEL"], fg=C["MUTED"],
                 anchor="w").pack(fill="x", padx=16)

        # ── separator ──────────────────────────────────────────────────────
        tk.Frame(self.res_outer, bg=C["BORDER"], height=1
                 ).pack(fill="x", padx=16, pady=6)

        # ── table ──────────────────────────────────────────────────────────
        tbl = tk.Frame(self.res_outer, bg=C["PANEL"])
        tbl.pack(fill="x", padx=16, pady=(0, 10))
        tbl.columnconfigure(3, weight=1)

        # header row
        for ci, htxt in enumerate(
                ["Bucket", "Unique Pairs", "% of All Pairs", "Bar chart"]):
            tk.Label(tbl, text=htxt,
                     font=("Segoe UI", 9, "bold"),
                     bg=C["BG"], fg=C["ACCENT"],
                     anchor="center", padx=12, pady=6
                     ).grid(row=0, column=ci,
                            sticky="nsew", padx=1, pady=1)

        # data rows: four buckets + walk-only + unreachable
        table_rows = (
            list(zip(BUCKET_LABELS,
                     [res["buckets"][b] for b in BUCKETS],
                     B_COLORS))
            + [("Walk Only",   res["walk_only"],   C["ACCENT"]),
               ("Unreachable", res["unreachable"],  C["RED"])]
        )

        for ri, (label, count, color) in enumerate(table_rows):
            bg = C["CARD"] if ri % 2 == 0 else C["PANEL"]
            pv = pct(count)

            # category
            tk.Label(tbl, text=label,
                     font=("Segoe UI", 10, "bold"),
                     bg=bg, fg=color,
                     anchor="w", padx=12, pady=7
                     ).grid(row=ri+1, column=0,
                            sticky="nsew", padx=1, pady=1)
            # count
            tk.Label(tbl, text=f"{count:,}",
                     font=("Courier New", 10),
                     bg=bg, fg=C["TEXT"],
                     anchor="center"
                     ).grid(row=ri+1, column=1,
                            sticky="nsew", padx=1, pady=1)
            # percentage
            tk.Label(tbl, text=f"{pv:.2f}%",
                     font=("Courier New", 10),
                     bg=bg, fg=C["TEXT"],
                     anchor="center"
                     ).grid(row=ri+1, column=2,
                            sticky="nsew", padx=1, pady=1)
            # bar
            bar_outer = tk.Frame(tbl, bg=bg, padx=8, pady=8)
            bar_outer.grid(row=ri+1, column=3,
                           sticky="nsew", padx=1, pady=1)
            bar_w = max(1, int(pv / 100 * 220)) if pv > 0 else 0
            bar_inner = tk.Frame(bar_outer, bg=bg)
            bar_inner.pack(fill="x")
            if bar_w:
                tk.Frame(bar_inner, bg=color,
                         width=bar_w, height=14
                         ).pack(side="left")
            tk.Label(bar_inner,
                     text=f"  {pv:.1f}%",
                     font=("Segoe UI", 8),
                     bg=bg, fg=color
                     ).pack(side="left")

        self.prog["value"] = 100
        self.lbl_prog.config(text="Analysis complete.")
        self.btn_run.config(state="normal", text="▶  Run Analysis")
        self.btn_export.config(state="normal")

    # ─── export ───────────────────────────────────────────────────────────────

    def _export(self):
        if not self._results:
            return
        res     = self._results
        total_p = res["total_possible"]
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="accessibility_summary.csv",
            title="Save CSV")
        if not path:
            return

        def pv(n):
            return f"{n / total_p * 100:.4f}" if total_p else "0"

        rows = []
        for bucket, label in zip(BUCKETS, BUCKET_LABELS):
            c = res["buckets"][bucket]
            rows.append({"Category": label,
                         "Unique_Pairs": c,
                         "Pct_of_Total": pv(c)})
        rows.append({"Category": "Walk Only",
                     "Unique_Pairs": res["walk_only"],
                     "Pct_of_Total": pv(res["walk_only"])})
        rows.append({"Category": "Transit Reachable (any bucket)",
                     "Unique_Pairs": res["transit_any"],
                     "Pct_of_Total": pv(res["transit_any"])})
        rows.append({"Category": "Unreachable",
                     "Unique_Pairs": res["unreachable"],
                     "Pct_of_Total": pv(res["unreachable"])})
        rows.append({"Category": "Total Possible",
                     "Unique_Pairs": total_p,
                     "Pct_of_Total": "100"})
        rows.append({"Category": "Total Hexes",
                     "Unique_Pairs": res["total_hexes"],
                     "Pct_of_Total": ""})

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["Category", "Unique_Pairs", "Pct_of_Total"])
            w.writeheader()
            w.writerows(rows)
        messagebox.showinfo("Saved", f"Results exported to:\n{path}")


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    App().mainloop()