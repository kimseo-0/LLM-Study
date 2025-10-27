# -*- coding: utf-8 -*-
"""
Custom PDF Loader (PyMuPDF)
- 위치 기반 클러스터링(왼쪽 → 위 우선) 텍스트 정렬
- 헤더/푸터 비율 크롭
- 이미지 추출 (stream / raster_clip / both)
  * OCR 없이, 이미지 bbox와 교차/근접한 텍스트로 설명(description) 생성
- 표(table):
  * PyMuPDF page.find_tables()로 탐지
  * 표 영역은 텍스트 추출에서 제외(중복 방지)
  * 표 이미지를 저장 (raster_clip)
  * 표 데이터를 Markdown으로 추출 (table.to_markdown())
- 앵커(Anchor) 옵션:
  * 페이지 텍스트에 [[FIG:...]], [[TAB:...]] 라인을 삽입(설명 포함)하도록 선택적 지원
- 반환: texts, clusters_info, images_info, tables_info
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import fitz  # PyMuPDF


# =========================
# 파라미터
# =========================
DEFAULT_PARAMS = dict(
    # 본문 클리핑
    header_ratio=0.078,     # 상단 크롭 비율
    footer_ratio=0.063,     # 하단 크롭 비율

    # 클러스터링/정렬
    v_overlap_min=0.35,
    h_gap_max_ratio=0.04,
    linf_max_ratio=0.03,
    row_merge_tol=0.6,
    x_bucket=8.0,           # 좌→우 정렬 안정화를 위한 x 버킷 폭(pt)

    # 표 탐지/저장
    table_min_cols=2,
    table_min_rows=2,
    table_raster_scale=2.0,

    # 이미지 설명 생성(텍스트 레이어 기반)
    fig_text_margin=12.0,   # bbox 주변 margin(pt) 안 텍스트 포함
    fig_caption_vdist=28.0, # 캡션 후보 세로 거리(아래/위)
    fig_caption_hcover=0.45,# 수평 커버 비율 임계(0~1), bbox와 같은 줄로 볼 최소 겹침
    fig_desc_maxlen=140,    # 설명 최대 길이

    # 앵커 라인 삽입 옵션
    anchor_embed_description=True  # [[FIG:...]]/[[TAB:...]] 라인 본문에 추가
)


# =========================
# 유틸
# =========================
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def rect_width(r):  return r[2] - r[0]
def rect_height(r): return r[3] - r[1]

def v_overlap_ratio(a, b) -> float:
    y0 = max(a[1], b[1])
    y1 = min(a[3], b[3])
    inter = max(0.0, y1 - y0)
    return inter / max(1e-6, min(rect_height(a), rect_height(b)))

def h_gap(a, b) -> float:
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return -min(a[2], b[2]) + max(a[0], b[0])

def dist_linf(a, b) -> float:
    dx = max(0.0, max(a[0]-b[2], b[0]-a[2]))
    dy = max(0.0, max(a[1]-b[3], b[1]-a[2]))
    return max(dx, dy)

def rects_intersect(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

def clip_inside_body(r, top_clip, bot_clip):
    # 세로 범위가 본문(body)와 교차하는지
    return not (r[3] < top_clip or r[1] > bot_clip)


# =========================
# 텍스트 추출/정렬
# =========================
def extract_spans(page: fitz.Page, top_clip: float, bot_clip: float):
    """page.get_text('dict')에서 span 단위 bbox/텍스트 수집."""
    d = page.get_text("dict")
    spans = []
    for b in d.get("blocks", []):
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                x0, y0, x1, y1 = sp["bbox"]
                if y1 < top_clip or y0 > bot_clip:
                    continue
                txt = sp.get("text", "").strip()
                if not txt:
                    continue
                spans.append(((x0, y0, x1, y1), txt))
    return spans

def cluster_spans(
    spans: List[Tuple[Tuple[float, float, float, float], str]],
    page_w: float,
    page_h: float,
    v_overlap_min: float,
    h_gap_max_ratio: float,
    linf_max_ratio: float
):
    """인접/유사 위치 스팬을 클러스터로 묶기(Union-Find)."""
    n = len(spans)
    if n == 0:
        return []
    dsu = DSU(n)
    h_gap_max = page_w * h_gap_max_ratio
    linf_max = max(page_w, page_h) * linf_max_ratio

    rects = [s[0] for s in spans]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rects[i], rects[j]
            if dist_linf(a, b) > linf_max:
                continue
            if v_overlap_ratio(a, b) >= v_overlap_min or h_gap(a, b) <= h_gap_max:
                dsu.union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)

    clusters = []
    for _, idxs in groups.items():
        clusters.append([spans[k] for k in idxs])
    return clusters

def sort_cluster_items(
    items: List[Tuple[Tuple[float, float, float, float], str]],
    row_merge_tol: float = 0.6
):
    """클러스터 내부 정렬: 행(위→아래) → 행 내 좌→우."""
    heights = [rect_height(r) for r, _ in items]
    base_h = sorted(heights)[len(heights)//2] if heights else 1.0
    y_tol = max(2.0, base_h * row_merge_tol)

    rows: List[List[Tuple[Tuple[float, float, float, float], str]]] = []
    items_sorted = sorted(items, key=lambda it: (it[0][1], it[0][0]))

    for it in items_sorted:
        r, _ = it
        placed = False
        for row in rows:
            y_avg = sum(x[0][1] for x in row) / len(row)
            if abs(r[1] - y_avg) <= y_tol:
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])

    for row in rows:
        row.sort(key=lambda it: it[0][0])
    rows.sort(key=lambda row: sum(x[0][1] for x in row) / len(row))

    ordered = []
    for row in rows:
        ordered.extend(row)
    return ordered

def representative_leftfirst_bucketed(items, x_bucket: float = 8.0):
    """클러스터 간 정렬 키: 왼쪽(x) 우선, 같은 x라인 내 위(y) 우선. x는 버킷팅으로 안정화."""
    x_min = min(r[0] for r, _ in items)
    y_min = min(r[1] for r, _ in items)
    x_bucketed = int(x_min // x_bucket)
    return (x_bucketed, y_min)


# =========================
# 표(Table) 처리
# =========================
def find_tables_on_page(
    page: fitz.Page,
    top_clip: float,
    bot_clip: float,
    min_cols: int = 2,
    min_rows: int = 2
):
    """
    PyMuPDF의 page.find_tables() 사용.
    반환: [{'id': 'TAB-pppp-iii', 'bbox':(x0,y0,x1,y1),'ncols':..,'nrows':..,'markdown': str}, ...]
    """
    tables_info = []
    try:
        found = page.find_tables()
        # 번호는 뒤에서 부여(페이지 로컬 카운터)
        for t in found.tables:
            x0, y0, x1, y1 = t.bbox
            if not clip_inside_body((x0, y0, x1, y1), top_clip, bot_clip):
                continue
            ncols = t.col_count
            nrows = t.row_count
            if ncols >= min_cols and nrows >= min_rows:
                md = t.to_markdown()
                tables_info.append({
                    # 'id':  -> 이후 assign
                    "bbox": (x0, y0, x1, y1),
                    "ncols": int(ncols),
                    "nrows": int(nrows),
                    "markdown": md
                })
    except Exception:
        pass
    return tables_info

def save_table_clips(
    page: fitz.Page,
    page_idx: int,
    tables_info: List[dict],
    out_dir: str,
    scale: float = 2.0
):
    """각 표 bbox를 렌더링해서 이미지로 저장 + id 부여."""
    ensure_dir(out_dir)
    saved = []
    for i, t in enumerate(tables_info, start=1):
        x0, y0, x1, y1 = t["bbox"]
        rect = fitz.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
        fn = os.path.join(out_dir, f"p{page_idx+1:04d}_t{i:03d}_table.png")
        pix.save(fn)
        rec = dict(t)
        rec["id"] = f"TAB-{page_idx+1:04d}-{i:03d}"
        rec["image_path"] = fn
        rec["image_scale"] = scale
        saved.append(rec)
    # id 없이 넘겨왔으면 여기서도 id 채움
    if not saved and tables_info:
        for i, t in enumerate(tables_info, start=1):
            t["id"] = f"TAB-{page_idx+1:04d}-{i:03d}"
        return tables_info
    return saved

def filter_spans_excluding_tables(
    spans: List[Tuple[Tuple[float, float, float, float], str]],
    table_bboxes: List[Tuple[float, float, float, float]]
):
    """표 bbox와 교차하는 스팬을 제거(표 텍스트 중복 방지)."""
    if not table_bboxes:
        return spans
    kept = []
    for r, txt in spans:
        if any(rects_intersect(r, tb) for tb in table_bboxes):
            continue
        kept.append((r, txt))
    return kept


# =========================
# 이미지 처리 (텍스트 레이어 기반 설명)
# =========================
def collect_image_bboxes_from_rawdict(page: fitz.Page, top_clip: float, bot_clip: float):
    """rawdict에서 이미지 블록(type==1) bbox와 xref 수집."""
    out = []
    d = page.get_text("rawdict")
    for b in d.get("blocks", []):
        if b.get("type") == 1:  # image block
            x0, y0, x1, y1 = b["bbox"]
            if y1 < top_clip or y0 > bot_clip:
                continue
            out.append({"bbox": (x0, y0, x1, y1), "xref": b.get("number")})
    return out

def spans_in_rect(spans, rect):
    """rect와 교차하는 스팬만 반환. rect: (x0,y0,x1,y1)"""
    x0,y0,x1,y1 = rect
    out=[]
    for (rx0,ry0,rx1,ry1), txt in spans:
        if not txt:
            continue
        if not (rx1 <= x0 or x1 <= rx0 or ry1 <= y0 or y1 <= ry0):
            out.append(((rx0,ry0,rx1,ry1), txt))
    return out

def horiz_cover_ratio(a, b):
    """수평 커버 비율: 두 구간의 교집합/작은쪽 길이"""
    ax0, ax1 = a
    bx0, bx1 = b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    denom = max(1e-6, min(ax1-ax0, bx1-bx0))
    return inter / denom

def _cluster_lines(spans_sorted):
    """간단 y-군집화로 라인 후보 묶기."""
    lines=[]; cur=[]
    for it in spans_sorted:
        if not cur:
            cur=[it]; continue
        y = it[0][1]
        y_avg = sum(s[0][1] for s in cur)/len(cur)
        # 행 높이 기반 허용치
        h_avg = sum((s[0][3]-s[0][1]) for s in cur)/len(cur)
        if abs(y - y_avg) <= max(2.0, h_avg*0.6):
            cur.append(it)
        else:
            lines.append(cur); cur=[it]
    if cur: lines.append(cur)
    return lines

def join_spans_linewise(spans):
    """y(상단)→x 순 정렬 후 같은 줄끼리 묶어 한 줄 문자열로 합침."""
    if not spans:
        return ""
    spans_sorted = sorted(spans, key=lambda it: (it[0][1], it[0][0]))
    lines = _cluster_lines(spans_sorted)
    texts=[]
    for line in lines:
        line = sorted(line, key=lambda it: it[0][0])
        texts.append(" ".join(t for _,t in line).strip())
    return " ".join(texts).strip()

def describe_image_from_text(
    spans, bbox, page_rect,
    margin=12.0, vdist=28.0, hcover=0.45, maxlen=140
):
    """
    이미지 설명을 페이지 텍스트 레이어에서 추정:
      1) bbox 내부 + margin 확장 박스의 텍스트
      2) 아래쪽 자막줄(수평 커버↑, 세로 거리 vdist 이내) → 위쪽 자막줄
      3) 없으면 가장 가까운 라인
    """
    x0,y0,x1,y1 = bbox

    # 1) 내부/주변 텍스트
    inner = spans_in_rect(spans, (x0-margin, y0-margin, x1+margin, y1+margin))
    if inner:
        s = join_spans_linewise(inner)
        if s:
            return (s[:maxlen-1]+"…") if len(s)>maxlen else s

    # 2) 아래/위 캡션 라인
    spans_sorted = sorted(spans, key=lambda it: (it[0][1], it[0][0]))
    lines = _cluster_lines(spans_sorted)

    bx = (x0, x1)
    by_center = (y0 + y1) * 0.5
    # 아래쪽 우선 → 위쪽
    best = None
    for pass_dir in ("below","above"):
        for line in lines:
            lx0 = min(r[0] for r,_ in line); lx1 = max(r[2] for r,_ in line)
            ly0 = min(r[1] for r,_ in line); ly1 = max(r[3] for r,_ in line)
            ly_center = (ly0+ly1)/2.0
            if pass_dir=="below" and ly_center < y1:  # 아래만
                continue
            if pass_dir=="above" and ly_center > y0:  # 위만
                continue
            vgap = min(abs(ly0 - y1), abs(ly1 - y0), abs(ly_center - by_center))
            if horiz_cover_ratio((lx0,lx1), bx) >= hcover and vgap <= vdist:
                text = " ".join(t for _,t in sorted(line, key=lambda it: it[0][0])).strip()
                if text:
                    cand = (vgap, text)
                    if (best is None) or (cand[0] < best[0]):
                        best = cand
        if best:
            text = best[1]
            return (text[:maxlen-1]+"…") if len(text)>maxlen else text

    # 3) 가장 가까운 라인
    if lines:
        lines.sort(key=lambda L: abs(((min(r[1] for r,_ in L)+max(r[3] for r,_ in L))/2.0) - by_center))
        text = " ".join(t for _,t in sorted(lines[0], key=lambda it: it[0][0])).strip()
        if text:
            return (text[:maxlen-1]+"…") if len(text)>maxlen else text

    return ""

def extract_images_from_page(
    doc: fitz.Document,
    page: fitz.Page,
    page_idx: int,
    out_dir: str,
    top_clip: float,
    bot_clip: float,
    mode: str = "stream",       # "stream" | "raster_clip" | "both"
    scale: float = 2.0,         # raster_clip 렌더 배율
    min_wh: int = 24,
    spans_for_desc=None,        # 본문 스팬(표 제외/헤더푸터 제외)
    page_rect=None,
    desc_margin: float = 12.0,
    desc_vdist: float = 28.0,
    desc_hcover: float = 0.45,
    desc_maxlen: int = 140
):
    """이미지 추출 + 텍스트 레이어 기반 설명 생성."""
    ensure_dir(out_dir)
    img_blocks = collect_image_bboxes_from_rawdict(page, top_clip, bot_clip)

    saved = []
    for idx, ib in enumerate(img_blocks, start=1):
        x0, y0, x1, y1 = ib["bbox"]
        if (x1 - x0) < min_wh or (y1 - y0) < min_wh:
            continue

        rec = {
            "id": f"FIG-{page_idx+1:04d}-{idx:03d}",
            "page": page_idx + 1,
            "bbox": (x0, y0, x1, y1),
            "width": x1 - x0,
            "height": y1 - y0
        }

        # 1) 내장 스트림
        if mode in ("stream", "both") and ib.get("xref"):
            try:
                img = doc.extract_image(ib["xref"])
                ext = img.get("ext", "png")
                fn = os.path.join(out_dir, f"p{page_idx+1:04d}_i{idx:03d}_stream.{ext}")
                with open(fn, "wb") as f:
                    f.write(img["image"])
                rec["stream_path"] = fn
                rec["stream_ext"] = ext
                rec["stream_xref"] = ib["xref"]
                rec["stream_w"] = img.get("width")
                rec["stream_h"] = img.get("height")
            except Exception:
                pass

        # 2) bbox 렌더링(보이는 그대로)
        if mode in ("raster_clip", "both"):
            clip_rect = fitz.Rect(x0, y0, x1, y1)
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
            fn = os.path.join(out_dir, f"p{page_idx+1:04d}_i{idx:03d}_clip.png")
            pix.save(fn)
            rec["clip_path"] = fn
            rec["clip_scale"] = scale

        # 3) 설명 생성(텍스트 레이어 기반)
        rec["description"] = ""
        if spans_for_desc and page_rect:
            desc = describe_image_from_text(
                spans=spans_for_desc, bbox=(x0,y0,x1,y1), page_rect=page_rect,
                margin=desc_margin, vdist=desc_vdist, hcover=desc_hcover, maxlen=desc_maxlen
            )
            rec["description"] = desc

        saved.append(rec)
    return saved


# =========================
# 앵커 라인 삽입(선택)
# =========================
def format_page_text_with_anchors(text: str, images_meta: List[dict], tables_meta: List[dict], params: Dict) -> str:
    """페이지 텍스트에 [[FIG:...]]/[[TAB:...]] 라인을 (설명/요약 포함) 추가."""
    if not params.get("anchor_embed_description", True):
        return text
    lines = []

    # 이미지 앵커
    for im in images_meta:
        fig_id = im.get("id")
        desc = (im.get("description") or "").strip()
        if fig_id:
            if desc:
                lines.append(f"[[FIG:{fig_id}]] {desc}")
            else:
                lines.append(f"[[FIG:{fig_id}]]")

    # 표 앵커 (마크다운 미니 요약: 헤더 줄만 짧게)
    for tb in tables_meta:
        tab_id = tb.get("id")
        md = (tb.get("markdown") or "").strip()
        mini = ""
        if md:
            # 첫 줄(헤더)만 요약
            head = md.splitlines()[0].strip()
            mini = f" {head}" if head else ""
        if tab_id:
            lines.append(f"[[TAB:{tab_id}]]{mini}")

    if lines:
        return (text.rstrip() + "\n\n" + "\n".join(lines)).strip()
    return text


# =========================
# 페이지 처리
# =========================
def process_page(
    page: fitz.Page,
    p: Dict,
    remove_repeat_headers: Optional[List[str]] = None,
    remove_repeat_footers: Optional[List[str]] = None,
    image_out_dir: Optional[str] = None,
    image_mode: Optional[str] = None,
    raster_scale: float = 2.0,
    table_out_dir: Optional[str] = None
):
    """페이지 단위 처리: (표 처리 포함) 텍스트 + 이미지 + 테이블."""
    w, h = page.rect.width, page.rect.height
    top_clip = h * p["header_ratio"]
    bot_clip = h * (1 - p["footer_ratio"])

    # 1) 표 탐지 + MD 추출
    tables_raw = find_tables_on_page(
        page, top_clip, bot_clip,
        min_cols=p.get("table_min_cols", 2),
        min_rows=p.get("table_min_rows", 2)
    )
    # 표 이미지 저장 + id 부여
    if table_out_dir:
        tables_meta = save_table_clips(
            page, page.number, tables_raw, table_out_dir,
            scale=p.get("table_raster_scale", 2.0)
        )
    else:
        # id 없으면 부여
        tables_meta = []
        if tables_raw:
            for i, t in enumerate(tables_raw, start=1):
                t["id"] = f"TAB-{page.number+1:04d}-{i:03d}"
                tables_meta.append(t)

    table_bboxes = [t["bbox"] for t in tables_meta] if tables_meta else []

    # 2) 텍스트 스팬 추출 → 표 영역 제외
    spans = extract_spans(page, top_clip, bot_clip)
    spans = filter_spans_excluding_tables(spans, table_bboxes)

    # 3) 텍스트 클러스터링/정렬
    clusters_sorted = []
    text = ""
    if spans:
        clusters = cluster_spans(
            spans, w, h,
            v_overlap_min=p["v_overlap_min"],
            h_gap_max_ratio=p["h_gap_max_ratio"],
            linf_max_ratio=p["linf_max_ratio"]
        )
        clusters_sorted_inside = [sort_cluster_items(c, row_merge_tol=p["row_merge_tol"]) for c in clusters]
        clusters_sorted = sorted(
            clusters_sorted_inside,
            key=lambda items: representative_leftfirst_bucketed(items, x_bucket=p["x_bucket"])
        )

        parts = []
        for items in clusters_sorted:
            seg = " ".join(txt for _, txt in items)
            parts.append(seg.strip())
        text = "\n".join([seg for seg in parts if seg])

        # 반복 헤더/푸터 문자열 제거(옵션)
        if remove_repeat_headers:
            for patt in remove_repeat_headers:
                text = re.sub(re.escape(patt) + r"\s*\n?", "", text)
        if remove_repeat_footers:
            for patt in remove_repeat_footers:
                text = re.sub(re.escape(patt) + r"\s*\n?", "", text)

    # 4) 일반 이미지 추출(텍스트 기반 설명 생성)
    images_meta = []
    if image_out_dir and image_mode:
        images_meta = extract_images_from_page(
            doc=page.parent, page=page, page_idx=page.number,
            out_dir=image_out_dir, top_clip=top_clip, bot_clip=bot_clip,
            mode=image_mode, scale=raster_scale,
            spans_for_desc=spans,                  # 표 제외/헤더푸터 제외 후 스팬
            page_rect=page.rect,
            desc_margin=p.get("fig_text_margin", 12.0),
            desc_vdist=p.get("fig_caption_vdist", 28.0),
            desc_hcover=p.get("fig_caption_hcover", 0.45),
            desc_maxlen=p.get("fig_desc_maxlen", 140)
        )

    # 5) 앵커 라인 삽입(옵션)
    text = format_page_text_with_anchors(text, images_meta, tables_meta, p)

    return text, clusters_sorted, images_meta, tables_meta


# =========================
# 전체 파일 처리 (Public API)
# =========================
def extract_pdf_with_cluster_order(
    path: str,
    detect_repeating: bool = False,     # 단순화: 반복 헤더/푸터 탐지 비활성 권장
    params: Optional[dict] = None,
    save_images: bool = False,
    images_out_dir: str = "./extracted_images",
    image_mode: str = "stream",         # "stream" | "raster_clip" | "both"
    raster_scale: float = 2.0,
    save_tables: bool = True,
    tables_out_dir: str = "./extracted_tables"
):
    """
    Returns:
        texts: List[str]                   # 페이지별 정렬된 텍스트(표 영역 제외, 앵커 라인 포함 가능)
        clusters_info: List[dict]          # 페이지별 클러스터 정보 요약
        images_info: List[dict]            # 페이지별 일반 이미지 메타 (id/bbox/paths/description)
        tables_info: List[dict]            # 페이지별 표 메타 (id/bbox/ncols/nrows/markdown/이미지 경로)
    """
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)

    doc = fitz.open(path)

    header_cands, footer_cands = None, None  # 필요 시 확장

    texts: List[str] = []
    clusters_info: List[dict] = []
    images_info: List[dict] = []
    tables_info: List[dict] = []

    for i in range(len(doc)):
        text, clusters, images_meta, tables_meta = process_page(
            doc[i], p=p,
            remove_repeat_headers=header_cands,
            remove_repeat_footers=footer_cands,
            image_out_dir=(images_out_dir if save_images else None),
            image_mode=(image_mode if save_images else None),
            raster_scale=raster_scale,
            table_out_dir=(tables_out_dir if save_tables else None)
        )
        texts.append(text)
        clusters_info.append({
            "page": i + 1,
            "num_clusters": len(clusters),
            "topleft": representative_leftfirst_bucketed(clusters[0], x_bucket=p["x_bucket"]) if clusters else None
        })
        images_info.append({
            "page": i + 1,
            "images": images_meta
        })
        tables_info.append({
            "page": i + 1,
            "tables": tables_meta
        })

    return texts, clusters_info, images_info, tables_info


# =========================
# 예시 실행
# =========================
if __name__ == "__main__":
    file_path = "./data/CN7N_2026_ko_KR.pdf"

    texts, info, images, tables = extract_pdf_with_cluster_order(
        file_path,
        detect_repeating=False,
        save_images=True,
        images_out_dir="./images_CN7N_2026",
        image_mode="both",     # "stream" | "raster_clip" | "both"
        raster_scale=2.0,
        save_tables=True,
        tables_out_dir="./tables_CN7N_2026"
    )
    print(info[:2])              # 클러스터 요약
    print(tables[:1])            # 1페이지 테이블 메타( markdown 포함 )
    print(images[:1])            # 1페이지 이미지 메타( description 포함 )
    print(texts[0][:300])        # 1페이지 텍스트 미리보기
