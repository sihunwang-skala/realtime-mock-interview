"""Text extraction for uploaded materials: format is detected from the bytes, transcripts lose their timestamps."""
import io
from qa_lab.core.extract import extract

def test_srt_keeps_words_drops_timestamps():
    r = extract("talk.srt", "1\n00:00:01,000 --> 00:00:03,000\n안녕하세요\n\n2\n00:00:03,500 --> 00:00:06,000\n예산은 삼십이억입니다\n".encode())
    assert r.fmt == "srt" and r.doc_type == "발표 전사" and "-->" not in r.text and "예산은 삼십이억입니다" in r.text

def test_html_drops_script_and_tags():
    r = extract("page.html", b"<html><script>x()</script><body><h1>Title</h1><p>Body &amp; text</p></body></html>")
    assert r.fmt == "html" and "x()" not in r.text and "Body & text" in r.text

def test_json_segments_joined():
    r = extract("n.json", b'{"segments":[{"text":" first"},{"text":" second"}]}')
    assert r.fmt == "json" and r.text == "first second"

def test_pptx_slides_and_notes():
    from pptx import Presentation
    p = Presentation(); s = p.slides.add_slide(p.slide_layouts[1]); s.shapes.title.text = "Budget"; s.notes_slide.notes_text_frame.text = "labour ratio"
    buf = io.BytesIO(); p.save(buf)
    r = extract("deck.pptx", buf.getvalue())
    assert r.fmt == "pptx" and r.doc_type == "발표자료" and "[슬라이드 1]" in r.text and "labour ratio" in r.text

def test_unknown_binary_gives_a_warning_not_a_crash():
    r = extract("x.bin", bytes(range(256)) * 4)
    assert isinstance(r.text, str)

def test_image_says_ocr_missing():
    r = extract("scan.png", b"\x89PNG\r\n\x1a\n" + b"0" * 50)
    assert r.fmt == "image" and r.warnings and "OCR" in r.warnings[0]
