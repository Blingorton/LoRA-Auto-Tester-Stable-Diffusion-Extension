"""
Auto-Test LoRAs — Stable Diffusion Reforge Extension  v9
"""
from __future__ import annotations
import os, json, shutil, hashlib, tempfile, re, threading, time
import gradio as gr
from modules import script_callbacks, shared

# ── Module-level stop flag ────────────────────────────────────────────────────
_STOP_EVENT = threading.Event()

LORA_EXT    = {".safetensors", ".pt", ".ckpt", ".bin"}
SIDECAR_EXT = [".json", ".png", ".info", ".civitai.info", ".preview.png"]
WC_SUBDIR   = os.path.join("extensions", "sd-dynamic-prompts", "wildcards", "autoloratest")
HARDCODED   = dict(width=1024, height=1024, steps=15, cfg=3.0,
                   sampler="Euler a", neg="~~illnp~~")
OUT_SUBDIR  = "Auto-Test LoRAs Outputs"

RETEST_STRONG    = "retest - too strong"
RETEST_WEAK      = "retest - too weak"
RETEST_PROMPT    = "retest - prompt tweak"
RETEST_SUBFOLDERS = {RETEST_STRONG, RETEST_WEAK, RETEST_PROMPT}

SORT_OPTIONS = ["Name A→Z", "Name Z→A", "Folder", "Has preview first"]

# ── Path helpers ──────────────────────────────────────────────────────────────

def get_lora_root():
    try:
        from modules import paths; return os.path.join(paths.models_path, "Lora")
    except: return os.path.join("models", "Lora")

def get_wc_root():
    try:
        from modules import paths; return os.path.join(paths.script_path, WC_SUBDIR)
    except: return WC_SUBDIR

def get_webui_root():
    try:
        from modules import paths; return paths.script_path
    except: return "."

def get_defaults():
    d = dict(HARDCODED)
    try:
        p = os.path.join(get_webui_root(), "ui-config.json")
        if os.path.exists(p):
            cfg = json.load(open(p, encoding="utf-8"))
            for k, dk in [("txt2img/Width/value","width"),
                          ("txt2img/Height/value","height"),
                          ("txt2img/Sampling steps/value","steps"),
                          ("txt2img/CFG Scale/value","cfg"),
                          ("txt2img/Sampler/value","sampler")]:
                if k in cfg: d[dk] = cfg[k]
    except: pass
    return d

# ── LoRA scanning & sorting ───────────────────────────────────────────────────

def scan_loras(root):
    out = []
    if not os.path.isdir(root): return out
    for dp, _, files in os.walk(root):
        rel   = os.path.relpath(dp, root).replace("\\", "/")
        parts = [p for p in rel.split("/") if p and p != "."]
        main  = parts[0] if parts else ""
        in_u  = any(p.lower() == "unused" for p in parts)
        in_t  = any(p.lower() == "tested" for p in parts)
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in LORA_EXT: continue
            fp   = os.path.join(dp, f).replace("\\", "/")
            stem = os.path.splitext(f)[0]
            prev = next((os.path.join(dp, stem+e).replace("\\", "/")
                         for e in [".preview.png", ".png"]
                         if os.path.exists(os.path.join(dp, stem+e))), None)
            out.append(dict(name=stem, path=fp, dir=dp.replace("\\","/"),
                            rel_dir=rel, main_sub=main,
                            unused=in_u, tested=in_t, preview=prev))
    return out

def sort_loras(loras, sort_by):
    if sort_by == "Name Z→A":
        return sorted(loras, key=lambda l: l["name"].lower(), reverse=True)
    elif sort_by == "Folder":
        return sorted(loras, key=lambda l: (l["rel_dir"].lower(), l["name"].lower()))
    elif sort_by == "Has preview first":
        return sorted(loras, key=lambda l: (0 if l["preview"] else 1, l["name"].lower()))
    else:  # Name A→Z (default)
        return sorted(loras, key=lambda l: l["name"].lower())

# ── Activation text ───────────────────────────────────────────────────────────

def get_activation_raw(path):
    """Full raw activation text; strips <network:...> tags but keeps || intact."""
    stem = os.path.splitext(path)[0]
    for ext in [".json", ".civitai.info"]:
        p = stem + ext
        if not os.path.exists(p): continue
        try:
            d = json.load(open(p, encoding="utf-8", errors="ignore"))
            w = d.get("trainedWords", [])
            raw = ", ".join(w) if w else d.get("activation text", d.get("activation_text", ""))
            if raw:
                raw = re.sub(r"<[^>]+>", "", raw)
                raw = re.sub(r",\s*,+", ",", raw).strip().strip(",").strip()
                return raw
        except: pass
    return ""

def get_activation(path, truncate=True):
    raw = get_activation_raw(path)
    if not raw: return ""
    if "||" not in raw: return raw
    if truncate:
        return raw.split("||")[0].rstrip(", ").strip()
    joined = re.sub(r"\s*\|\|\s*", ", ", raw)
    return re.sub(r",\s*,", ",", joined).strip().strip(",").strip()

# ── Prompt files ──────────────────────────────────────────────────────────────

def get_prompt_files(sub, wc_root):
    if not sub or not os.path.isdir(wc_root): return []
    pre = sub.lower() + "_"
    out = []
    try:
        for f in sorted(os.listdir(wc_root)):
            if not f.lower().endswith(".txt") or not f.lower().startswith(pre): continue
            tag = f[len(pre):-4]
            try: content = open(os.path.join(wc_root, f), encoding="utf-8", errors="ignore").read()
            except: content = ""
            out.append(dict(tag=tag, content=content))
    except: pass
    return out

# ── Move helpers ──────────────────────────────────────────────────────────────

def _move_files(stem, src_dir, dst_dir, extra_imgs=None, img_suffix=""):
    os.makedirs(dst_dir, exist_ok=True)
    for ext in SIDECAR_EXT:
        s = os.path.join(src_dir, stem + ext)
        if os.path.exists(s):
            d = os.path.join(dst_dir, stem + ext)
            if not os.path.exists(d):
                try: shutil.move(s, d)
                except: pass
    for ext in LORA_EXT:
        s = os.path.join(src_dir, stem + ext)
        if os.path.exists(s):
            d = os.path.join(dst_dir, stem + ext)
            if not os.path.exists(d):
                try: shutil.move(s, d)
                except: pass
    if extra_imgs:
        for src_img, base_name in extra_imgs:
            if not os.path.exists(src_img): continue
            if img_suffix:
                n, e = os.path.splitext(base_name)
                dest_name = f"{n}{img_suffix}{e}"
            else:
                dest_name = base_name
            d = os.path.join(dst_dir, dest_name)
            if not os.path.exists(d):
                try: shutil.move(src_img, d)
                except: pass

def move_lora_to_tested(lpath, extra_imgs=None, img_suffix=""):
    ldir     = os.path.dirname(lpath)
    category = os.path.dirname(ldir)
    td       = os.path.join(category, "tested")
    created  = not os.path.isdir(td)
    if created:
        for sub in ["good", "retest - prompt tweak", "retest - too strong", "retest - too weak"]:
            os.makedirs(os.path.join(td, sub), exist_ok=True)
    stem = os.path.splitext(os.path.basename(lpath))[0]
    _move_files(stem, ldir, td, extra_imgs, img_suffix)
    return td

def move_lora_from_retest(lpath, extra_imgs=None, img_suffix=""):
    ldir   = os.path.dirname(lpath)
    tested = os.path.dirname(ldir)
    stem   = os.path.splitext(os.path.basename(lpath))[0]
    _move_files(stem, ldir, tested, extra_imgs, img_suffix)
    return tested

# ── Wildcard resolution & prompt building ────────────────────────────────────

def resolve_wildcards(prompt):
    try:
        from dynamicprompts.generators import RandomPromptGenerator
        r = RandomPromptGenerator().generate(prompt, count=1)
        return r[0] if r else prompt
    except: pass
    try:
        prompt = re.sub(r"\{([^}]+)\}", lambda m: m.group(1).split("|")[0].strip(), prompt)
    except: pass
    return prompt

def build_prompt(name, act, tmpl, strength=1.0):
    lstr = f"<lora:{name}:{strength:.2f}>"
    if act: lstr = act + ", " + lstr
    base = (tmpl.replace("*lora*", lstr) if tmpl and "*lora*" in tmpl
            else (tmpl + ", " + lstr) if tmpl else lstr)
    return resolve_wildcards(base)

def save_image_with_metadata(img, path, prompt):
    try:
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Comment", prompt)
        img.save(path, pnginfo=meta)
    except Exception:
        img.save(path)

# ── Placeholder thumbnail ─────────────────────────────────────────────────────

def _placeholder(name):
    key = hashlib.md5(name.encode()).hexdigest()
    tmp = os.path.join(tempfile.gettempdir(), f"atlora_{key}.png")
    if os.path.exists(tmp): return tmp
    try:
        import struct, zlib
        def chunk(tag, data):
            c = tag + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        idat = zlib.compress(b"\x00\x22\x22\x22")
        open(tmp, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR",ihdr) + chunk(b"IDAT",idat) + chunk(b"IEND",b""))
        return tmp
    except: return None

def loras_to_gallery(loras):
    out = []
    for l in loras:
        img = l["preview"] if (l["preview"] and os.path.isfile(l["preview"])) else _placeholder(l["name"])
        if not img: continue
        badge = " [U]" if l["unused"] else (" [T]" if l["tested"] else "")
        out.append((img, l["name"] + badge))
    return out

# ── Filter / folder helpers ───────────────────────────────────────────────────

def filter_loras(loras, filt, folder):
    out = loras
    if folder and folder != "(all)":
        out = [l for l in out if l["rel_dir"] == folder]
    if filt and filt.strip():
        fl = filt.strip().lower()
        out = [l for l in out if fl in l["name"].lower()]
    return out

def folder_choices(loras):
    from collections import Counter
    counts = Counter(l["rel_dir"] for l in loras)
    return ["(all)"] + [f"{f}  ({counts[f]})" for f in sorted(counts)]

def folder_from_display(s):
    return s.split("  (")[0] if "  (" in s else s

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _sel_info_html(paths, loras):
    if not paths:
        return ('<div class="at-selinfo">Click thumbnail then "+ Add" to queue. '
                'Select a folder to queue all LoRAs in it.</div>')
    lmap = {l["path"].replace("\\","/"): l for l in loras}
    parts = []
    for p in paths[:4]:
        l = lmap.get(p.replace("\\","/"))
        if l:
            b = (' <span class="at-bu">U</span>' if l["unused"] else
                 ' <span class="at-bt">T</span>' if l["tested"] else "")
            parts.append(f'<span class="at-sn">{l["name"]}{b}</span>')
    extra = f' <span class="at-dim">+{len(paths)-4} more</span>' if len(paths) > 4 else ""
    return (f'<div class="at-selinfo"><b>{len(paths)}</b> selected: '
            + " ".join(parts) + extra + "</div>")

def _act_display_html(raw_act, truncate, ignore=False):
    if ignore:
        if not raw_act:
            return '<div class="at-actbox at-actbox-empty">(no activation text)</div>'
        raw_esc = raw_act.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        return f'<div class="at-actbox"><span class="at-struck">{raw_esc}</span> <span style="color:#f87171;font-size:10px">[ignored]</span></div>'
    if not raw_act:
        return '<div class="at-actbox at-actbox-empty">(no activation text)</div>'
    raw_esc = raw_act.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    if "||" not in raw_act or not truncate:
        return f'<div class="at-actbox">{raw_esc}</div>'
    keep, _, rest = raw_esc.partition("||")
    struck = f'<span class="at-struck">||{rest}</span>'
    return f'<div class="at-actbox">{keep}{struck}</div>'

def _preview_html(pfs):
    if not pfs: return ""
    rows = ""
    for pf in pfs:
        snippet = (pf["content"][:500]
                   .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        rows += (f'<details style="margin:2px 0;border-radius:4px;border:1px solid #1e1e2e;'
                 f'padding:3px 8px;background:#050510">'
                 f'<summary style="cursor:pointer;color:#888;font-size:11px;'
                 f'list-style:none;user-select:none;padding:2px 0">'
                 f'<span style="color:#a78bfa;font-weight:700">{pf["tag"]}</span>'
                 f' &nbsp;&#9660; preview</summary>'
                 f'<pre style="font-size:10px;color:#666;white-space:pre-wrap;'
                 f'font-family:monospace;margin:4px 0 2px;max-height:80px;'
                 f'overflow-y:auto">{snippet}</pre></details>')
    return f'<div style="margin-top:4px">{rows}</div>'

def _queue_html(queue):
    if not queue:
        return '<div class="at-queuebox at-queuebox-empty">(no jobs queued)</div>'
    rows = ""
    for i, job in enumerate(queue):
        sel = json.loads(job.get("sel_json","[]"))
        names = ", ".join(os.path.splitext(os.path.basename(p))[0] for p in sel[:3])
        if len(sel) > 3: names += f" +{len(sel)-3} more"
        tags = ", ".join(job.get("checked_tags") or ["(all)"])
        rows += (f'<div class="at-qjob"><span class="at-qnum">#{i+1}</span> '
                 f'<b>{names}</b> &nbsp;<span class="at-qdim">str={job.get("lora_strength",1.0):.2f} '
                 f'tags=[{tags}]</span></div>')
    return f'<div class="at-queuebox">{rows}</div>'

# ── script_args ───────────────────────────────────────────────────────────────

def _get_script_args():
    import modules.scripts as mscripts
    scripts_runner = mscripts.scripts_txt2img
    all_scripts = getattr(scripts_runner, "alwayson_scripts", [])
    if not all_scripts: return []
    max_to = max((s.args_to for s in all_scripts), default=0)
    if max_to <= 0: return []
    live = None
    try: live = list(getattr(scripts_runner, "script_args", None) or [])
    except: pass
    if not live:
        try:
            inputs = getattr(scripts_runner, "inputs", None) or []
            if inputs: live = [getattr(c, "value", False) for c in inputs]
        except: pass
    if live and len(live) >= max_to:
        return list(live[:max_to])
    args = [False] * max_to
    for script in all_scripts:
        lo, hi = script.args_from, script.args_to
        n = hi - lo
        if n <= 0: continue
        fname = os.path.basename(getattr(script, "filename", "") or "").lower()
        if "seed" in fname:
            args[lo:hi] = ([False, False, 0, 0] + [False]*(n-4))[:n]
        elif "forge_never_oom" in fname or ("never" in fname and "oom" in fname):
            args[lo:hi] = ([True, True] + [False]*(n-2))[:n]
        elif "hypertile" in fname:
            args[lo:hi] = ([False, 256, 2, 4] + [False]*(n-4))[:n]
        elif "kohya" in fname:
            args[lo:hi] = ([False, 2, 1.0] + [False]*(n-3))[:n]
        elif "refiner" in fname:
            args[lo:hi] = ([False, "", 0.8] + [False]*(n-3))[:n]
        elif "controlnet" in fname:
            try:
                from scripts.controlnet import ControlNetUnit
                args[lo:hi] = [ControlNetUnit(enabled=False)] * n
            except: pass
    return args

# ── Core generation (single job) ─────────────────────────────────────────────

def _run_job(job, overall_done, overall_total, progress_fn=None):
    """
    Run one job dict. Yields (imgs_so_far, log_lines, overall_frac, lora_frac,
    overall_desc, lora_desc) tuples as work proceeds.
    Returns final (imgs, log_str).
    """
    sel_json     = job["sel_json"]
    tc_json      = job["tc_json"]
    checked_tags = job["checked_tags"]
    strength     = float(job.get("lora_strength", 1.0))
    truncate_act = bool(job.get("truncate_act", True))
    ignore_act   = bool(job.get("ignore_act", False))
    w, h         = int(job["w"]), int(job["h"])
    steps, cfg   = int(job["steps"]), float(job["cfg"])
    sampler      = job["sampler"]
    seed         = int(job["seed"])
    neg          = job["neg"]

    try:    sel = json.loads(sel_json)
    except: sel = []
    try:    tc  = json.loads(tc_json)
    except: tc  = {}
    if not sel: return [], "No LoRAs selected."

    try:
        from modules import processing, devices
        from modules.processing import StableDiffusionProcessingTxt2Img, fix_seed
        import modules.scripts as mscripts
    except Exception as e:
        return [], f"Import error: {e}"

    try:
        base_out = shared.opts.outdir_txt2img_samples or shared.opts.outdir_samples or ""
        base_out = os.path.abspath(base_out) if base_out else os.path.join(get_webui_root(), "outputs", "txt2img-images")
    except:
        base_out = os.path.join(get_webui_root(), "outputs", "txt2img-images")
    outpath = os.path.join(base_out, OUT_SUBDIR)
    os.makedirs(outpath, exist_ok=True)

    try:    script_args = _get_script_args()
    except: script_args = None

    log, imgs = [], []
    tags      = checked_tags if checked_tags else [None]
    job_total = len(sel) * len(tags)

    for lora_idx, lpath in enumerate(sel):
        if _STOP_EVENT.is_set():
            log.append("-- Stopped by user --")
            break

        lname        = os.path.splitext(os.path.basename(lpath))[0]
        act          = "" if ignore_act else get_activation(lpath, truncate=truncate_act)
        ldir         = os.path.dirname(lpath)
        parent_lower = os.path.basename(ldir).lower()
        in_u         = parent_lower == "unused"
        retest_t     = parent_lower if parent_lower in RETEST_SUBFOLDERS else ""

        LR_norm  = get_lora_root().replace("\\", "/").rstrip("/")
        lp_norm  = lpath.replace("\\", "/")
        rel      = lp_norm[len(LR_norm):].lstrip("/")
        rel_parts = [p for p in rel.split("/") if p]
        category = rel_parts[0] if rel_parts and rel_parts[0].lower() not in ("unused","tested") else "misc"
        cat_out  = os.path.join(outpath, category)
        os.makedirs(cat_out, exist_ok=True)

        lora_imgs = []

        for tag_idx, tag in enumerate(tags):
            tmpl = tc.get(tag) if tag else None
            pmt  = build_prompt(lname, act, tmpl, strength)
            tlbl = tag if tag else "default"

            # Progress fractions
            job_done      = lora_idx * len(tags) + tag_idx
            ov_frac       = (overall_done + job_done) / max(overall_total, 1)
            lora_frac     = tag_idx / max(len(tags), 1)
            ov_desc       = f"{lname} [{lora_idx+1}/{len(sel)}]  tag: {tlbl}"
            lora_desc     = f"template {tag_idx+1}/{len(tags)}: {tlbl}"

            try:
                p = StableDiffusionProcessingTxt2Img(
                    sd_model=shared.sd_model, prompt=pmt, negative_prompt=neg,
                    width=w, height=h, steps=steps, cfg_scale=cfg,
                    sampler_name=sampler, seed=seed,
                    do_not_save_grid=True, do_not_save_samples=True,
                    outpath_samples=cat_out, outpath_grids=cat_out,
                )
                p.scripts    = mscripts.scripts_txt2img
                p.script_args = list(script_args) if script_args is not None else []
                p.steps        = steps
                p.sampler_name = sampler
                p.cfg_scale    = cfg
                fix_seed(p)
                with devices.autocast():
                    proc = processing.process_images(p)

                img = proc.images[0] if proc.images else None
                if img:
                    outf = f"{lname}_{tlbl}.png"
                    save_image_with_metadata(img, os.path.join(cat_out, outf), pmt)
                    save_image_with_metadata(img, os.path.join(ldir, outf), pmt)
                    imgs.append(img)
                    lora_imgs.append((outf, os.path.join(ldir, outf)))
                    log.append(f"OK  {lname}/{tlbl}")
                else:
                    log.append(f"WARN {lname}/{tlbl} -> no image")
            except Exception as e:
                import traceback; traceback.print_exc()
                log.append(f"ERR  {lname}/{tlbl} -> {e}")

            yield imgs, log, ov_frac, lora_frac, ov_desc, lora_desc

        # Move after all templates for this LoRA
        extra = [(src, base) for base, src in lora_imgs]
        if in_u and lora_imgs:
            move_lora_to_tested(lpath, extra_imgs=extra)
            log.append(f"  -> moved {lname} to tested/")
        elif retest_t and lora_imgs:
            if retest_t == RETEST_PROMPT:
                suffix = "_retested with full activation text"
            else:
                suffix = f"_retested at {strength:.2f}"
            move_lora_from_retest(lpath, extra_imgs=extra, img_suffix=suffix)
            log.append(f"  -> moved {lname} from {retest_t}/ back to tested/")

    # Refresh LoRA cache
    try:
        from modules import extra_networks
        extra_networks.clear_extra_network_cache()
    except: pass
    try:
        from modules.networks import load_networks
        load_networks.cache.clear()
    except: pass

    return imgs, "\n".join(log)

# ── Tab ───────────────────────────────────────────────────────────────────────

def build_tab():
    LR  = get_lora_root()
    WCR = get_wc_root()
    DEF = get_defaults()

    try:    all_loras = scan_loras(LR)
    except Exception as e:
        print(f"[Auto-Test LoRAs] scan error: {e}"); all_loras = []

    try:
        from modules import samplers as S
        slist = [s.name for s in S.all_samplers]
    except:
        slist = ["Euler a", "Euler", "DPM++ 2M Karras", "DDIM", "UniPC"]
    if DEF["sampler"] not in slist:
        DEF["sampler"] = slist[0] if slist else "Euler a"

    fc = folder_choices(all_loras)

    with gr.Blocks(css=CSS) as tab:
        st_all     = gr.State(all_loras)
        st_filt    = gr.State(all_loras)
        st_sel     = gr.State("[]")
        st_tc      = gr.State("{}")
        st_pfs     = gr.State([])
        st_last    = gr.State("")
        st_sort    = gr.State("Name A→Z")
        st_truncate   = gr.State(True)
        st_ignore_act = gr.State(False)
        st_queue   = gr.State([])   # list of job dicts

        gr.HTML('<div class="at-hdr">Auto-Test LoRAs '
                '<span style="font-size:12px;font-weight:400;color:#555">v9</span></div>')

        with gr.Row(equal_height=False):
            # ── LEFT: LoRA browser ───────────────────────────────────────
            with gr.Column(scale=5):
                with gr.Row():
                    filt_box    = gr.Textbox(placeholder="Filter by name...",
                                             show_label=False, scale=4)
                    refresh_btn = gr.Button("↺ Refresh", scale=1, size="sm")

                with gr.Row():
                    sort_dd = gr.Dropdown(
                        choices=SORT_OPTIONS, value="Name A→Z",
                        label="Sort", scale=3)

                folder_radio = gr.Radio(
                    choices=fc, value="(all)",
                    label="Folder  (select to queue all LoRAs in it)",
                    elem_classes=["at-folderradio"],
                )

                sel_info = gr.HTML(_sel_info_html([], all_loras))
                with gr.Row():
                    clear_btn = gr.Button("✕ Clear", size="sm", elem_classes=["at-clearbtn"])
                    add_btn   = gr.Button("+ Add to selection", size="sm", elem_classes=["at-addbtn"])
                gr.HTML('<div class="at-hint">Click thumbnail, then "+ Add" to queue '
                        '&nbsp;|&nbsp; Select folder to queue all</div>')

                lora_gallery = gr.Gallery(
                    value=None, show_label=False,
                    columns=5, height=1200,
                    object_fit="cover", allow_preview=False,
                    elem_classes=["at-gallery"], elem_id="at_gallery_inner",
                )

            # ── RIGHT: settings + run ─────────────────────────────────────
            with gr.Column(scale=6):
                gr.HTML('<div class="at-sec">Activation Text</div>')
                act_display = gr.HTML('<div class="at-actbox at-actbox-empty">(select a LoRA)</div>')
                with gr.Row():
                    truncate_toggle = gr.Button(
                        "✂ Truncate at ||  (ON)", size="sm",
                        elem_id="at_trunc_btn",
                    )
                    ignore_act_btn = gr.Button(
                        "⊘ Ignore Act. Text  (OFF)", size="sm",
                        elem_id="at_ignore_btn",
                    )

                gr.HTML('<hr class="at-hr"/><div class="at-sec">Prompt Templates</div>')
                with gr.Row():
                    sel_all_btn   = gr.Button("☑ Select All",   size="sm", elem_classes=["at-selbtn"])
                    desel_all_btn = gr.Button("☐ Deselect All", size="sm", elem_classes=["at-deselbtn"])
                tag_check    = gr.CheckboxGroup(choices=[], value=[], label="",
                                                elem_classes=["at-tagcheck"])
                tpl_previews = gr.HTML("")

                gr.HTML('<hr class="at-hr"/>')
                lora_strength = gr.Slider(minimum=-5, maximum=5, step=0.05, value=1.0,
                                          label="LoRA Strength")

                gr.HTML('<div class="at-sec">Generation Settings</div>')
                with gr.Row():
                    w_sl = gr.Slider(minimum=64, maximum=2048, step=64,
                                     value=DEF["width"],  label="Width")
                    h_sl = gr.Slider(minimum=64, maximum=2048, step=64,
                                     value=DEF["height"], label="Height")
                with gr.Row():
                    st_sl = gr.Slider(minimum=1, maximum=150, step=1,
                                      value=DEF["steps"], label="Steps")
                    cf_sl = gr.Slider(minimum=1, maximum=30,  step=0.5,
                                      value=DEF["cfg"],   label="CFG Scale")
                with gr.Row():
                    smp  = gr.Dropdown(choices=slist, value=DEF["sampler"], label="Sampler")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                neg = gr.Textbox(value=DEF["neg"], lines=3, label="Negative Prompt")
                cpneg_btn = gr.Button("Copy neg from txt2img tab", size="sm")

                gr.HTML('<hr class="at-hr"/>')
                # Queue panel
                gr.HTML('<div class="at-sec">Test Queue</div>')
                queue_display = gr.HTML(_queue_html([]))
                with gr.Row():
                    enqueue_btn = gr.Button("+ Add to Queue", size="sm",
                                            elem_classes=["at-addbtn"])
                    clear_queue_btn = gr.Button("✕ Clear Queue", size="sm",
                                                elem_classes=["at-clearbtn"])

                gr.HTML('<hr class="at-hr"/>')
                with gr.Row():
                    run_btn  = gr.Button("▶ Run", variant="primary", scale=3)
                    run_queue_btn = gr.Button("▶ Run Queue", variant="primary", scale=3)
                    stop_btn = gr.Button("⏹ Stop after current", scale=2, size="sm",
                                         elem_classes=["at-stopbtn"])

                # Progress bars
                prog_overall = gr.HTML('<div class="at-progbar-wrap" style="display:none"></div>')
                prog_lora    = gr.HTML('<div class="at-progbar-wrap" style="display:none"></div>')

                out_gal = gr.Gallery(label="Generated Images", columns=4, height=300)
                log_box = gr.Textbox(label="Log", lines=8, interactive=False)

        # ── Helpers ──────────────────────────────────────────────────────

        def _prog_html(frac, label, color="#818cf8"):
            pct = max(0.0, min(1.0, frac)) * 100
            return (f'<div class="at-progbar-wrap">'
                    f'<div class="at-progbar-label">{label}</div>'
                    f'<div class="at-progbar-track">'
                    f'<div class="at-progbar-fill" style="width:{pct:.1f}%;background:{color}"></div>'
                    f'</div><div class="at-progbar-pct">{pct:.0f}%</div></div>')

        def _upd(paths, loras, prev_checked=None):
            if not paths:
                return ("[]","{}",[], gr.update(value=_sel_info_html([],loras)),
                        gr.update(choices=[],value=[]), "")
            p0 = paths[0].replace("\\", "/")
            first = next((l for l in loras
                          if l["path"].replace("\\","/") == p0), None)
            # fallback: match by filename stem only
            if first is None:
                stem0 = os.path.splitext(os.path.basename(p0))[0]
                first = next((l for l in loras
                              if l["name"] == stem0), None)
            sub    = first["main_sub"] if first else ""
            pfs    = get_prompt_files(sub, WCR)
            tc     = {pf["tag"]: pf["content"] for pf in pfs}
            checked = prev_checked if prev_checked is not None else list(tc.keys())
            print(f"[Auto-Test LoRAs] _upd: paths={paths}, sub={sub!r}, templates={list(tc.keys())}")
            return (json.dumps(paths), json.dumps(tc), pfs,
                    gr.update(value=_sel_info_html(paths, loras)),
                    gr.update(choices=list(tc.keys()), value=checked),
                    _preview_html(pfs))

        def _retest_overrides(paths, loras):
            if not paths: return gr.update(), gr.update(), True
            l = next((x for x in loras if x["path"] == paths[0]), None)
            if not l: return gr.update(), gr.update(), True
            parent = os.path.basename(l["dir"]).lower()
            if parent == RETEST_STRONG:
                return gr.update(value=0.8), gr.update(value="✂ Truncate at ||  (ON)"), True
            elif parent == RETEST_WEAK:
                return gr.update(value=1.2), gr.update(value="✂ Truncate at ||  (ON)"), True
            elif parent == RETEST_PROMPT:
                return gr.update(value=1.0), gr.update(value="✂ Truncate at ||  (OFF)"), False
            else:
                return gr.update(value=1.0), gr.update(value="✂ Truncate at ||  (ON)"), True

        def _snapshot(sel_json, tc_json, checked_tags, strength, truncate, ignore,
                      w, h, steps, cfg, sampler, seed, neg):
            return dict(sel_json=sel_json, tc_json=tc_json,
                        checked_tags=checked_tags, lora_strength=float(strength),
                        truncate_act=bool(truncate), ignore_act=bool(ignore),
                        w=w, h=h, steps=steps,
                        cfg=cfg, sampler=sampler, seed=seed, neg=neg)

        # ── Callbacks ────────────────────────────────────────────────────

        def cb_load():
            d = get_defaults()
            return (loras_to_gallery(all_loras),
                    gr.update(value=d["width"]),  gr.update(value=d["height"]),
                    gr.update(value=d["steps"]),  gr.update(value=d["cfg"]),
                    gr.update(value=d["neg"]))
        tab.load(fn=cb_load, outputs=[lora_gallery, w_sl, h_sl, st_sl, cf_sl, neg])

        def cb_refresh():
            loras = scan_loras(LR); d = get_defaults(); fc2 = folder_choices(loras)
            return (loras, loras, loras_to_gallery(loras),
                    gr.update(choices=fc2, value="(all)"),
                    gr.update(value=d["width"]),  gr.update(value=d["height"]),
                    gr.update(value=d["steps"]),  gr.update(value=d["cfg"]),
                    gr.update(value=d["neg"]))
        refresh_btn.click(fn=cb_refresh,
            outputs=[st_all, st_filt, lora_gallery, folder_radio,
                     w_sl, h_sl, st_sl, cf_sl, neg])

        def cb_filter_sort(filt, folder_disp, sort_by, loras):
            folder   = folder_from_display(folder_disp) if folder_disp != "(all)" else ""
            filtered = filter_loras(loras, filt, folder)
            filtered = sort_loras(filtered, sort_by)
            return filtered, loras_to_gallery(filtered)

        filt_box.change(fn=cb_filter_sort,
            inputs=[filt_box, folder_radio, sort_dd, st_all],
            outputs=[st_filt, lora_gallery])

        sort_dd.change(fn=cb_filter_sort,
            inputs=[filt_box, folder_radio, sort_dd, st_all],
            outputs=[st_filt, lora_gallery])

        # Clicking a thumbnail immediately selects that single LoRA (replaces
        # current selection) AND remembers it in st_last so "+ Add" can append it.
        def cb_click(evt: gr.SelectData, filt_loras, loras, truncate, ignore):
            idx = evt.index
            if idx < 0 or idx >= len(filt_loras):
                return ("", gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), True)
            lpath = filt_loras[idx]["path"]
            raw   = get_activation_raw(lpath)
            # Single-click = select just this one LoRA
            cur   = [lpath]
            upd   = _upd(cur, loras)
            str_upd, trunc_upd, new_trunc = _retest_overrides(cur, loras)
            act_upd = gr.update(value=_act_display_html(raw, new_trunc, ignore))
            return (lpath,) + upd + (str_upd, trunc_upd, act_upd, new_trunc)
        lora_gallery.select(fn=cb_click,
            inputs=[st_filt, st_all, st_truncate, st_ignore_act],
            outputs=[st_last, st_sel, st_tc, st_pfs, sel_info, tag_check, tpl_previews,
                     lora_strength, truncate_toggle, act_display, st_truncate])

        # "+ Add to selection": appends st_last to existing selection without replacing it.
        def cb_add(last_path, cur_json, loras, truncate, ignore):
            if not last_path:
                return (cur_json, gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), True)
            try:    cur = json.loads(cur_json)
            except: cur = []
            if last_path not in cur:
                cur = cur + [last_path]
            upd = _upd(cur, loras)
            str_upd, trunc_upd, new_trunc = _retest_overrides(cur, loras)
            raw = get_activation_raw(cur[0]) if cur else ""
            act_upd = gr.update(value=_act_display_html(raw, new_trunc, ignore))
            return upd + (str_upd, trunc_upd, act_upd, new_trunc)
        add_btn.click(fn=cb_add,
            inputs=[st_last, st_sel, st_all, st_truncate, st_ignore_act],
            outputs=[st_sel, st_tc, st_pfs, sel_info, tag_check, tpl_previews,
                     lora_strength, truncate_toggle, act_display, st_truncate])

        def cb_folder(folder_disp, filt, sort_by, loras):
            folder   = folder_from_display(folder_disp) if folder_disp != "(all)" else ""
            filtered = sort_loras(filter_loras(loras, filt, folder), sort_by)
            paths    = [l["path"] for l in filtered]
            upd = _upd(paths, loras)
            str_upd, trunc_upd, new_trunc = _retest_overrides(paths, loras)
            act_upd = gr.update(value=_act_display_html("", False))
            return (filtered, loras_to_gallery(filtered)) + upd + (str_upd, trunc_upd, act_upd, new_trunc)
        folder_radio.change(fn=cb_folder,
            inputs=[folder_radio, filt_box, sort_dd, st_all],
            outputs=[st_filt, lora_gallery,
                     st_sel, st_tc, st_pfs, sel_info, tag_check, tpl_previews,
                     lora_strength, truncate_toggle, act_display, st_truncate])

        def cb_clear(loras):
            return _upd([], loras) + ("", gr.update(),
                    gr.update(value="✂ Truncate at ||  (ON)"),
                    gr.update(value='<div class="at-actbox at-actbox-empty">(select a LoRA)</div>'),
                    True, False, gr.update(value="⊘ Ignore Act. Text  (OFF)"))
        clear_btn.click(fn=cb_clear, inputs=[st_all],
            outputs=[st_sel, st_tc, st_pfs, sel_info, tag_check, tpl_previews,
                     st_last, lora_strength, truncate_toggle, act_display, st_truncate, st_ignore_act, ignore_act_btn])

        def cb_trunc_toggle(cur_state, last_path, ignore):
            new_state = not cur_state
            label = "✂ Truncate at ||  (ON)" if new_state else "✂ Truncate at ||  (OFF)"
            raw   = get_activation_raw(last_path) if last_path else ""
            return new_state, gr.update(value=label), gr.update(value=_act_display_html(raw, new_state, ignore))
        truncate_toggle.click(fn=cb_trunc_toggle,
            inputs=[st_truncate, st_last, st_ignore_act],
            outputs=[st_truncate, truncate_toggle, act_display])

        def cb_ignore_toggle(cur_ignore, last_path, truncate):
            new_ignore = not cur_ignore
            label = "⊘ Ignore Act. Text  (ON)" if new_ignore else "⊘ Ignore Act. Text  (OFF)"
            raw   = get_activation_raw(last_path) if last_path else ""
            return new_ignore, gr.update(value=label), gr.update(value=_act_display_html(raw, truncate, new_ignore))
        ignore_act_btn.click(fn=cb_ignore_toggle,
            inputs=[st_ignore_act, st_last, st_truncate],
            outputs=[st_ignore_act, ignore_act_btn, act_display])

        def cb_sel_all(pfs):
            return gr.update(value=[pf["tag"] for pf in pfs])
        sel_all_btn.click(fn=cb_sel_all, inputs=[st_pfs], outputs=[tag_check])

        def cb_desel_all():
            return gr.update(value=[])
        desel_all_btn.click(fn=cb_desel_all, outputs=[tag_check])

        def cb_cpneg():
            try:
                v = getattr(shared, "txt2img_negative_prompt", None)
                return v if v else HARDCODED["neg"]
            except: return HARDCODED["neg"]
        cpneg_btn.click(fn=cb_cpneg, outputs=[neg])

        # Queue callbacks
        def cb_enqueue(sel_json, tc_json, checked_tags, strength, truncate,
                       w, h, steps, cfg, sampler, seed, neg, queue):
            job = _snapshot(sel_json, tc_json, checked_tags, strength, truncate, ignore,
                            w, h, steps, cfg, sampler, seed, neg)
            new_queue = queue + [job]
            return new_queue, gr.update(value=_queue_html(new_queue))
        enqueue_btn.click(fn=cb_enqueue,
            inputs=[st_sel, st_tc, tag_check, lora_strength, st_truncate, st_ignore_act,
                    w_sl, h_sl, st_sl, cf_sl, smp, seed, neg, st_queue],
            outputs=[st_queue, queue_display])

        def cb_clear_queue():
            return [], gr.update(value=_queue_html([]))
        clear_queue_btn.click(fn=cb_clear_queue, outputs=[st_queue, queue_display])

        # ── Run (single job, generator) ───────────────────────────────────
        def do_run(sel_json, tc_json, checked_tags, strength, truncate, ignore,
                   w, h, steps, cfg, sampler, seed, neg):
            _STOP_EVENT.clear()
            try:    sel = json.loads(sel_json)
            except: sel = []
            if not sel:
                yield ([], "No LoRAs selected. Click a thumbnail to select one.",
                       gr.update(), gr.update())
                return
            job = _snapshot(sel_json, tc_json, checked_tags, strength, truncate, ignore,
                            w, h, steps, cfg, sampler, seed, neg)
            total = max(len(sel) * len(checked_tags if checked_tags else [None]), 1)

            all_imgs, all_log = [], []
            try:
                gen = _run_job(job, 0, total)
                while True:
                    try:
                        imgs, log, ov_frac, lr_frac, ov_desc, lr_desc = next(gen)
                        all_imgs = imgs
                        all_log  = log
                        yield (all_imgs,
                               "\n".join(all_log),
                               gr.update(value=_prog_html(ov_frac, f"Overall: {ov_desc}")),
                               gr.update(value=_prog_html(lr_frac, f"LoRA: {lr_desc}", "#34d399")))
                    except StopIteration as e:
                        if e.value is not None:
                            final_imgs, final_log = e.value
                            all_imgs = final_imgs
                            all_log  = final_log.split("\n") if isinstance(final_log, str) else final_log
                        break
            except Exception as e:
                import traceback; traceback.print_exc()
                all_log.append(f"FATAL: {e}")

            yield (all_imgs, "\n".join(all_log) if isinstance(all_log, list) else all_log,
                   gr.update(value=_prog_html(1.0, "Done")),
                   gr.update(value=_prog_html(1.0, "Done", "#34d399")))

        run_btn.click(fn=do_run,
            inputs=[st_sel, st_tc, tag_check, lora_strength, st_truncate, st_ignore_act,
                    w_sl, h_sl, st_sl, cf_sl, smp, seed, neg],
            outputs=[out_gal, log_box, prog_overall, prog_lora],
            queue=True)

        # ── Run Queue (generator) ─────────────────────────────────────────
        def do_run_queue(queue):
            if not queue:
                yield [], "Queue is empty.", gr.update(), gr.update(), queue, gr.update()
                return
            _STOP_EVENT.clear()
            jobs = list(queue)

            # Count total images across all jobs
            grand_total = 0
            for job in jobs:
                try: sel = json.loads(job["sel_json"])
                except: sel = []
                grand_total += len(sel) * len(job.get("checked_tags") or [None])
            grand_total = max(grand_total, 1)

            all_imgs, all_log = [], []
            grand_done = 0

            for job_idx, job in enumerate(jobs):
                if _STOP_EVENT.is_set():
                    all_log.append("-- Stopped by user --")
                    break
                try:    sel = json.loads(job["sel_json"])
                except: sel = []
                job_total = len(sel) * len(job.get("checked_tags") or [None])

                try:
                    gen = _run_job(job, grand_done, grand_total)
                    while True:
                        try:
                            imgs, log, ov_frac, lr_frac, ov_desc, lr_desc = next(gen)
                            all_imgs = imgs
                            all_log  = log
                            remaining_jobs = jobs[job_idx+1:]
                            yield (all_imgs,
                                   "\n".join(all_log),
                                   gr.update(value=_prog_html(ov_frac, f"Job {job_idx+1}/{len(jobs)}: {ov_desc}")),
                                   gr.update(value=_prog_html(lr_frac, f"LoRA: {lr_desc}", "#34d399")),
                                   remaining_jobs,
                                   gr.update(value=_queue_html(remaining_jobs)))
                        except StopIteration as e:
                            if e.value is not None:
                                final_imgs, final_log = e.value
                                all_imgs = final_imgs
                                all_log  = final_log.split("\n") if isinstance(final_log, str) else final_log
                            break
                except Exception as e:
                    import traceback; traceback.print_exc()
                    all_log.append(f"FATAL job {job_idx+1}: {e}")

                grand_done += job_total

            empty_queue = []
            yield (all_imgs,
                   "\n".join(all_log) if isinstance(all_log, list) else all_log,
                   gr.update(value=_prog_html(1.0, f"Done — {len(jobs)} job(s) complete")),
                   gr.update(value=_prog_html(1.0, "Done", "#34d399")),
                   empty_queue,
                   gr.update(value=_queue_html(empty_queue)))

        run_queue_btn.click(fn=do_run_queue,
            inputs=[st_queue],
            outputs=[out_gal, log_box, prog_overall, prog_lora, st_queue, queue_display],
            queue=True)

        def cb_stop():
            _STOP_EVENT.set()
        stop_btn.click(fn=cb_stop)

    return [(tab, "Auto-Test LoRAs", "auto_test_loras")]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
.at-hdr{font-size:20px;font-weight:800;letter-spacing:.04em;
  padding:10px 0 8px;border-bottom:1px solid #333;margin-bottom:12px}
.at-sec{font-size:11px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:#818cf8;margin-bottom:4px;margin-top:10px}
.at-hr{border:none;border-top:1px solid #2e2e4e;margin:12px 0}
.at-dim{color:#666;font-size:12px;padding:4px 0}
.at-hint{color:#555;font-size:10px;margin-bottom:6px}
.at-folderradio .wrap{max-height:130px;overflow-y:auto}
.at-folderradio label{font-size:11px!important;padding:2px 6px!important}
.at-selinfo{background:rgba(129,140,248,.1);border-left:3px solid #818cf8;
  padding:6px 10px;border-radius:5px;font-size:12px;margin:4px 0;line-height:1.7}
.at-sn{margin-right:4px}
.at-bu{background:#7c3aed;color:#fff;border-radius:3px;font-size:8px;padding:1px 3px}
.at-bt{background:#059669;color:#fff;border-radius:3px;font-size:8px;padding:1px 3px}
.at-clearbtn{font-size:11px!important;color:#888!important}
.at-addbtn{font-size:11px!important;color:#a78bfa!important;border-color:#a78bfa!important}
.at-selbtn{font-size:11px!important;color:#34d399!important;border-color:#34d399!important}
.at-deselbtn{font-size:11px!important;color:#888!important}
.at-stopbtn{font-size:11px!important;color:#f87171!important;border-color:#f87171!important}

/* Activation text */
.at-actbox{background:#07070f;border:1px solid #1e1e38;border-radius:6px;
  padding:6px 10px;font-size:11px;color:#a0a0c0;font-family:monospace;
  white-space:pre-wrap;word-break:break-all;min-height:28px;margin-bottom:4px;line-height:1.5}
.at-actbox-empty{color:#444!important}
.at-struck{text-decoration:line-through;color:#555!important;opacity:0.7}

/* Truncate toggle */
#at_trunc_btn{font-size:11px!important;color:#f59e0b!important;border-color:#f59e0b!important}
#at_ignore_btn{font-size:11px!important;color:#f87171!important;border-color:#f87171!important}

/* Queue display */
.at-queuebox{background:#07070f;border:1px solid #1e1e38;border-radius:6px;
  padding:6px 8px;font-size:11px;min-height:28px;max-height:120px;overflow-y:auto}
.at-queuebox-empty{color:#444!important}
.at-qjob{padding:2px 0;border-bottom:1px solid #1e1e38;line-height:1.6}
.at-qjob:last-child{border-bottom:none}
.at-qnum{color:#818cf8;font-weight:700;margin-right:4px}
.at-qdim{color:#555;font-size:10px}

/* Progress bars */
.at-progbar-wrap{margin:4px 0 2px;font-size:10px;color:#888}
.at-progbar-label{margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.at-progbar-track{background:#1e1e38;border-radius:4px;height:8px;width:100%;overflow:hidden}
.at-progbar-fill{height:8px;border-radius:4px;transition:width .3s ease}
.at-progbar-pct{text-align:right;font-size:9px;color:#555;margin-top:1px}

/* Gallery portrait thumbnails */
#at_gallery_inner{
  height:1200px!important;min-height:200px!important;
  overflow-y:auto!important;overflow-x:hidden!important;
  resize:vertical!important;border:1px solid #2e2e4e!important;
  border-radius:8px!important;box-sizing:border-box!important}
#at_gallery_inner > div,
#at_gallery_inner .grid-container,
#at_gallery_inner [class*="svelte"]{
  height:auto!important;max-height:none!important;overflow:visible!important}
#at_gallery_inner .thumbnail-item,
#at_gallery_inner .gallery-item,
#at_gallery_inner li{
  width:80px!important;min-width:80px!important;max-width:80px!important;
  height:120px!important;min-height:120px!important;max-height:120px!important;
  overflow:hidden!important;border-radius:5px!important;
  margin:2px!important;padding:0!important;flex-shrink:0!important}
#at_gallery_inner .thumbnail-item button,
#at_gallery_inner .gallery-item button,
#at_gallery_inner li > button,
#at_gallery_inner li > div{
  width:80px!important;height:120px!important;
  padding:0!important;overflow:hidden!important;display:block!important}
#at_gallery_inner img{
  width:80px!important;height:120px!important;
  min-width:80px!important;min-height:120px!important;
  max-width:80px!important;max-height:120px!important;
  object-fit:cover!important;object-position:top center!important;display:block!important}
#at_gallery_inner .caption-label,
#at_gallery_inner figcaption,
#at_gallery_inner [class*="caption"]{
  font-size:7px!important;max-height:13px!important;overflow:hidden!important;
  white-space:nowrap!important;text-overflow:ellipsis!important}

/* Template CheckboxGroup */
.at-tagcheck .wrap{display:flex;flex-direction:column;gap:4px}
.at-tagcheck label{display:flex;align-items:center;gap:8px;
  border:2px solid #1e1e38;border-radius:6px;padding:6px 10px;background:#07070f;
  transition:border-color .15s,background .15s;cursor:pointer;margin:0!important}
.at-tagcheck label:has(input:checked){border-color:#7c3aed!important;background:#110720!important}
.at-tagcheck input[type=checkbox]{width:18px!important;height:18px!important;
  accent-color:#7c3aed;flex-shrink:0;cursor:pointer}
.at-tagcheck .wrap span,
.at-tagcheck span{font-size:12px!important;font-weight:700!important;color:#666!important}
.at-tagcheck label:has(input:checked) span{color:#f472b6!important}
"""

try:
    script_callbacks.on_ui_tabs(build_tab)
except Exception as e:
    import traceback
    print(f"[Auto-Test LoRAs] FATAL: {e}")
    traceback.print_exc()
