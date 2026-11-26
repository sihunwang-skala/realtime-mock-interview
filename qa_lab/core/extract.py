"""Text extraction for any uploaded material: proposal / RFP (PDF, DOCX, HWP-less text), presentation slides (PPTX), and speech transcripts (TXT, SRT, VTT, JSON).
Detects the format from the file's bytes (not only the name), returns plain text plus a short note for anything that could not be read."""
from __future__ import annotations
import html, io, json, os, re, shutil, subprocess, tempfile, zipfile
from dataclasses import dataclass, field

@dataclass
class Extracted:
    text: str = ""
    fmt: str = "unknown"                       # pdf | pptx | ppt | docx | doc | xlsx | html | srt | vtt | json | text | image | binary
    doc_type: str = "문서"                      # guessed: 제안서/RFP · 발표자료 · 발표 전사
    pages: int = 0
    warnings: list[str] = field(default_factory=list)

MAX_UPLOAD_BYTES = 60 * 1024 * 1024

def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp949", "euc-kr", "utf-8"):
        try: return raw.decode(enc)
        except Exception: continue
    return raw.decode("utf-8", errors="ignore")

def _sniff(name: str, raw: bytes) -> str:
    ext = os.path.splitext(name.lower())[1]
    if raw.startswith(b"%PDF"): return "pdf"
    if raw.startswith(b"PK\x03\x04"):
        try:
            names = zipfile.ZipFile(io.BytesIO(raw)).namelist()
        except Exception: return "binary"
        if any(n.startswith("ppt/slides/") for n in names): return "pptx"
        if any(n.startswith("word/") for n in names): return "docx"
        if any(n.startswith("xl/") for n in names): return "xlsx"
        return "zip"
    if raw.startswith(b"\xd0\xcf\x11\xe0"):              # old Office (OLE): .doc / .ppt / .xls / .hwp
        return {".ppt": "ppt", ".pps": "ppt", ".doc": "doc", ".hwp": "hwp"}.get(ext, "ole")
    if raw.startswith(b"\x89PNG") or raw[:3] == b"\xff\xd8\xff" or raw[:4] in (b"GIF8", b"II*\x00", b"MM\x00*"): return "image"
    if raw.startswith(b"{\\rtf"): return "rtf"
    head = raw[:4096].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if head.startswith((b"<!doctype html", b"<html", b"<?xml")) or ext in (".html", ".htm"): return "html"
    if ext == ".srt" or re.search(rb"^\d+\s*\r?\n\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->", raw[:4096], re.M): return "srt"
    if head.startswith(b"webvtt") or ext == ".vtt": return "vtt"
    if ext == ".json" or head.startswith((b"{", b"[")):
        try: json.loads(_decode(raw)); return "json"
        except Exception: pass
    return "text"

def _pdf(raw: bytes, out: Extracted) -> None:
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(raw)); out.pages = len(r.pages)
        parts = [(p.extract_text() or "").strip() for p in r.pages]
        out.text = "\n\n".join(p for p in parts if p)
    except Exception as ex:
        out.warnings.append(f"PDF 읽기 오류: {ex!r}"[:200])
    if len(out.text.strip()) < 20 and shutil.which("pdftotext"):          # fall back to poppler for PDFs that pypdf reads badly
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(raw); f.flush()
            res = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", f.name, "-"], capture_output=True, timeout=120)
            out.text = res.stdout.decode("utf-8", errors="ignore")
    if len(out.text.strip()) < 20:
        out.warnings.append("PDF에서 글자를 찾지 못했어요. 스캔한 사진 형태라면 글자 인식(OCR)이 필요합니다.")

def _pptx(raw: bytes, out: Extracted) -> None:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(raw)); out.pages = len(prs.slides); out.doc_type = "발표자료"
    chunks = []
    for i, slide in enumerate(prs.slides, 1):
        lines = []
        for shp in slide.shapes:
            if shp.has_text_frame: lines += [p.text.strip() for p in shp.text_frame.paragraphs if p.text.strip()]
            if getattr(shp, "has_table", False) and shp.has_table:
                for row in shp.table.rows: lines.append(" | ".join(c.text.strip() for c in row.cells))
            if shp.shape_type == 6:                                          # group: read its children too
                for sub in shp.shapes:
                    if sub.has_text_frame: lines += [p.text.strip() for p in sub.text_frame.paragraphs if p.text.strip()]
        note = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None else ""
        if lines or note:
            chunks.append(f"[슬라이드 {i}]\n" + "\n".join(lines) + (f"\n(발표자 노트) {note}" if note else ""))
    out.text = "\n\n".join(chunks)
    if not out.text.strip(): out.warnings.append("슬라이드에서 글자를 찾지 못했어요. 그림으로만 된 슬라이드일 수 있습니다.")

def _docx(raw: bytes, out: Extracted) -> None:
    import docx
    d = docx.Document(io.BytesIO(raw))
    parts = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows: parts.append(" | ".join(c.text.strip() for c in row.cells))
    out.text = "\n".join(parts)

def _xlsx(raw: bytes, out: Extracted) -> None:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True); chunks = []
    for ws in wb.worksheets:
        rows = [" | ".join("" if v is None else str(v) for v in r) for r in ws.iter_rows(values_only=True)]
        rows = [r for r in rows if r.replace("|", "").strip()]
        if rows: chunks.append(f"[시트 {ws.title}]\n" + "\n".join(rows))
    out.text = "\n\n".join(chunks)

def _textutil(raw: bytes, ext: str, out: Extracted) -> None:
    """Old .doc / .rtf / .odt: macOS textutil converts them to plain text."""
    if not shutil.which("textutil"): out.warnings.append("이 형식은 읽을 수 없어요. .docx나 .pdf로 저장해 주세요."); return
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in" + ext); open(src, "wb").write(raw)
        r = subprocess.run(["textutil", "-convert", "txt", "-stdout", src], capture_output=True, timeout=120)
        out.text = r.stdout.decode("utf-8", errors="ignore")
    if not out.text.strip(): out.warnings.append("글자를 찾지 못했어요.")

def _html(raw: bytes, out: Extracted) -> None:
    s = _decode(raw)
    s = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6]|section|article)>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    out.text = "\n".join(re.sub(r"[ \t]+", " ", html.unescape(l)).strip() for l in s.splitlines() if l.strip())

def _subtitle(raw: bytes, out: Extracted, vtt: bool) -> None:
    """Speech transcripts with timestamps: keep the words, drop the numbers and arrows."""
    lines = []
    for l in _decode(raw).splitlines():
        l = l.strip()
        if not l or l.upper().startswith(("WEBVTT", "NOTE")) or re.fullmatch(r"\d+", l) or "-->" in l: continue
        l = re.sub(r"<[^>]+>", "", l)
        if lines and lines[-1] == l: continue                                 # the same caption repeated
        lines.append(l)
    out.text = " ".join(lines); out.doc_type = "발표 전사"

def _json(raw: bytes, out: Extracted) -> None:
    d = json.loads(_decode(raw))
    if isinstance(d, dict) and isinstance(d.get("segments"), list):        # Whisper-style: {"text", "segments":[{"text"}]}
        out.text = " ".join(str(s.get("text", "")).strip() for s in d["segments"]); out.doc_type = "발표 전사"
    elif isinstance(d, dict) and isinstance(d.get("text"), str): out.text = d["text"]
    else:
        out.text = "\n".join(str(x) for x in d) if isinstance(d, list) else json.dumps(d, ensure_ascii=False, indent=1)

def extract(name: str, raw: bytes) -> Extracted:
    if len(raw) > MAX_UPLOAD_BYTES: return Extracted(fmt="binary", warnings=["파일이 너무 커요 (60MB 이하)."])
    fmt = _sniff(name, raw); out = Extracted(fmt=fmt); ext = os.path.splitext(name.lower())[1]
    try:
        if fmt == "pdf": _pdf(raw, out)
        elif fmt == "pptx": _pptx(raw, out)
        elif fmt == "ppt": out.warnings.append("구형 .ppt예요. PowerPoint에서 .pptx로 저장해서 올려 주세요.")
        elif fmt == "docx": _docx(raw, out)
        elif fmt in ("doc", "rtf"): _textutil(raw, ext or ("." + fmt), out)
        elif fmt == "xlsx": _xlsx(raw, out)
        elif fmt == "html": _html(raw, out)
        elif fmt in ("srt", "vtt"): _subtitle(raw, out, vtt=fmt == "vtt")
        elif fmt == "json": _json(raw, out)
        elif fmt == "text": out.text = _decode(raw)
        elif fmt == "image": out.warnings.append("이미지 속 글자 인식(OCR)은 아직 연결되지 않았어요. 글을 복사해서 붙여 주세요.")
        elif fmt == "hwp": out.warnings.append("한글(.hwp) 파일은 아직 읽을 수 없어요. PDF나 .docx로 저장해서 올려 주세요.")
        else: out.warnings.append("이 형식은 읽을 수 없어요.")
    except Exception as ex:
        out.warnings.append(f"파일을 읽는 중 오류: {ex!r}"[:200])
    out.text = re.sub(r"[ \t]+\n", "\n", out.text).strip()
    if fmt == "text" and out.text and len(re.findall(r"\d{2}:\d{2}:\d{2}", out.text)) > 5: out.doc_type = "발표 전사"
    return out
