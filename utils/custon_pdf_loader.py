import fitz  # PyMuPDF
import re
from typing import List, Tuple, Dict

# ---------- 유틸 ----------
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

def rect_center(rect):  # (x0,y0,x1,y1) -> (cx, cy)
    x0, y0, x1, y1 = rect
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5

def rect_width(rect):
    return rect[2] - rect[0]
def rect_height(rect):
    return rect[3] - rect[1]

def v_overlap_ratio(a, b):
    # 두 상자의 수직 겹침 비율 (겹침 높이 / min(height_a, height_b))
    y0 = max(a[1], b[1])
    y1 = min(a[3], b[3])
    inter = max(0.0, y1 - y0)
    return inter / max(1e-6, min(rect_height(a), rect_height(b)))

def h_gap(a, b):
    # 수평 간격(가로 축 기준, 음수면 겹침)
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return -min(a[2], b[2]) + max(a[0], b[0])

def dist_linf(a, b):
    # L-infinity 거리 (bbox 외곽 간격 기준 대략치)
    dx = max(0.0, max(a[0]-b[2], b[0]-a[2]))
    dy = max(0.0, max(a[1]-b[3], b[1]-a[3]))
    return max(dx, dy)

# ---------- 핵심 파라미터 ----------
PARAMS = dict(
    header_ratio = 0.078,   # 상단 6% 잘라내기
    footer_ratio = 0.063,   # 하단 6% 잘라내기
    v_overlap_min = 0.35,  # 같은 묶음으로 볼 최소 수직 겹침 비율
    h_gap_max_ratio = 0.04,# 페이지 폭 대비 허용 수평 간격 (가까우면 같은 묶음)
    linf_max_ratio = 0.03, # 페이지 대각선 대신 L∞ 거리(폭/높이 스케일) 임계
    row_merge_tol = 0.6,   # 같은 행으로 묶는 y-기준(행 내 정렬용)
)

# ---------- 스팬(문자 덩어리) 추출 ----------
def extract_spans(page, top_clip, bottom_clip):
    """
    page.get_text('dict')에서 span 단위 bbox와 텍스트를 수집.
    헤더/푸터 제외 높이(top_clip~bottom_clip)만 포함.
    return: [(rect, text), ...] where rect=(x0,y0,x1,y1)
    """
    d = page.get_text("dict")
    spans = []
    for b in d.get("blocks", []):
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                x0, y0, x1, y1 = sp["bbox"]
                if y1 < top_clip or y0 > bottom_clip:
                    continue
                txt = sp.get("text", "").strip()
                if not txt:
                    continue
                spans.append(((x0, y0, x1, y1), txt))
    return spans

# ---------- 클러스터링 ----------
def cluster_spans(spans: List[Tuple[Tuple[float,float,float,float], str]], page_w, page_h, 
                  v_overlap_min, h_gap_max_ratio, linf_max_ratio):
    """
    인접/유사 위치 스팬을 같은 클러스터로 묶는다(Union-Find).
    - 수직 겹침이 충분하거나
    - 수평 갭이 충분히 작거나
    - 전반적 거리(L∞)가 작으면 같은 묶음
    """
    n = len(spans)
    dsu = DSU(n)
    h_gap_max = page_w * h_gap_max_ratio
    linf_max = max(page_w, page_h) * linf_max_ratio

    rects = [s[0] for s in spans]

    # 간단 O(n^2). 문서가 크면 공간 인덱싱(kd-tree)로 교체 가능.
    for i in range(n):
        for j in range(i+1, n):
            a, b = rects[i], rects[j]
            # 빨리 거를 조건: 너무 멀면 skip
            if dist_linf(a, b) > linf_max:
                continue

            vo = v_overlap_ratio(a, b)
            hg = h_gap(a, b)

            if vo >= v_overlap_min or hg <= h_gap_max:
                dsu.union(i, j)

    # 클러스터별 모으기
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)

    # 클러스터를 [(indices, rect, text...), ...] 형태로 변환
    clusters = []
    for gid, idxs in groups.items():
        items = [spans[k] for k in idxs]
        clusters.append(items)
    return clusters

# ---------- 정렬 ----------
def sort_cluster_items(items: List[Tuple[Tuple[float,float,float,float], str]], row_merge_tol=0.6):
    """
    한 클러스터 내부 정렬:
      1) y(상단) 기준으로 행 만들고
      2) 행 내에서는 x(좌->우)로 정렬
    row_merge_tol: 같은 행으로 칠 y거리 임계 (행 높이 기반 비율)
    """
    # 라인 높이의 대표값(중앙값)을 잡자
    heights = [rect_height(r) for r, _ in items]
    base_h = sorted(heights)[len(heights)//2] if heights else 1.0
    y_tol = max(2.0, base_h * row_merge_tol)

    # 행 그룹핑
    rows: List[List[Tuple[Tuple[float,float,float,float], str]]] = []
    # items를 y0 기준으로 정렬
    items_sorted = sorted(items, key=lambda it: (it[0][1], it[0][0]))

    for it in items_sorted:
        r, t = it
        placed = False
        for row in rows:
            # 같은 행인지 판단: 기존 행의 평균 y0와 비교
            y_avg = sum(x[0][1] for x in row) / len(row)
            if abs(r[1] - y_avg) <= y_tol:
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])

    # 각 행 내 x정렬
    for row in rows:
        row.sort(key=lambda it: it[0][0])

    # 행 자체는 y 오름차순
    rows.sort(key=lambda row: sum(x[0][1] for x in row) / len(row))

    # 텍스트 병합
    ordered = []
    for row in rows:
        for it in row:
            ordered.append(it)
    return ordered

def representative_topleft(items):
    # 클러스터의 대표 좌상단 좌표(정렬 키)
    x_min = min(r[0] for r, _ in items)
    y_min = min(r[1] for r, _ in items)
    return (y_min, x_min)  # 먼저 위(y), 그다음 왼쪽(x)

def representative_leftfirst(items):
    # 클러스터의 대표 좌표: x_min 먼저, y_min 다음
    x_min = min(r[0] for r, _ in items)
    y_min = min(r[1] for r, _ in items)
    return (x_min, y_min)  # ← 왼쪽 우선

def representative_leftfirst_bucketed(items, x_bucket=8.0):
    x_min = min(r[0] for r, _ in items)
    y_min = min(r[1] for r, _ in items)
    # x를 버킷으로 묶어서 거의 같은 세로줄을 하나로 보이게
    x_bucketed = int(x_min // x_bucket)
    return (x_bucketed, y_min)

# ---------- 페이지 처리 ----------
def process_page(page, p=PARAMS, remove_repeat_headers=None, remove_repeat_footers=None):
    w, h = page.rect.width, page.rect.height
    top_clip = h * p["header_ratio"]
    bot_clip = h * (1 - p["footer_ratio"])

    spans = extract_spans(page, top_clip, bot_clip)
    if not spans:
        return "", []

    clusters = cluster_spans(
        spans, w, h,
        v_overlap_min=p["v_overlap_min"],
        h_gap_max_ratio=p["h_gap_max_ratio"],
        linf_max_ratio=p["linf_max_ratio"]
    )

    # 클러스터 내부 정렬
    clusters_sorted_inside = [sort_cluster_items(c, row_merge_tol=p["row_merge_tol"]) for c in clusters]
    # 클러스터 간 좌상단 우선 정렬(위→아래, 왼→오)
    clusters_sorted = sorted(clusters_sorted_inside, key=representative_leftfirst_bucketed)

    # 텍스트 합치기
    parts = []
    for items in clusters_sorted:
        seg = " ".join(txt for _, txt in items)
        parts.append(seg.strip())

    text = "\n".join([seg for seg in parts if seg])

    # 반복 헤더/푸터 문자열 추가 제거(옵션)
    if remove_repeat_headers:
        for patt in remove_repeat_headers:
            text = re.sub(re.escape(patt) + r"\s*\n?", "", text)
    if remove_repeat_footers:
        for patt in remove_repeat_footers:
            text = re.sub(re.escape(patt) + r"\s*\n?", "", text)

    # 빈 줄 정리
    # text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, clusters_sorted

# ---------- 전체 파일 ----------
def extract_pdf_with_cluster_order(
    path: str,
    detect_repeating: bool = False,
    params: dict = None
):
    if params is None:
        params = PARAMS

    doc = fitz.open(path)
    # (선택) 반복 헤더/푸터 문자열 후보 찾기 (간단 버전: 비활성 권장)
    header_cands, footer_cands = None, None
    if detect_repeating:
        # 필요하면 여기에 앞서 만든 반복탐지 루틴을 붙여도 됨
        header_cands, footer_cands = [], []

    texts = []
    debug_clusters = []

    for i in range(len(doc)):
        t, clusters = process_page(doc[i], p=params,
                                   remove_repeat_headers=header_cands,
                                   remove_repeat_footers=footer_cands)
        texts.append(t)
        debug_clusters.append({
            "page": i+1,
            "num_clusters": len(clusters),
            "topleft": representative_leftfirst_bucketed(clusters[0]) if clusters else None
        })

    return texts, debug_clusters


# ----------------- 사용 예시 -----------------
if __name__ == "__main__":
    file_path = './data/CN7N_2026_ko_KR.pdf'
    full, info = extract_pdf_with_cluster_order(file_path, detect_repeating=False)
    print(info[:5])
    # full을 RAG chunking -> embedding -> vector store 단계로 넘기면 됨

    print(full[39])
