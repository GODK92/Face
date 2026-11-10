"""AI 관상 분석 Streamlit 앱 (3단계: UI 디자인)."""

import base64
import datetime
import hashlib
import html
import io
import json
import os
import threading
from pathlib import Path

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

from prompts import FEATURE_KEYS, FORTUNE_KEYS, SYSTEM_PROMPT, USER_PROMPT

DEFAULT_MODEL = "gpt-4o"
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_SIDE = 1024
MAX_CALLS_PER_SESSION = 10
DEFAULT_DAILY_LIMIT = 200      # 앱 전체(모든 사용자) 하루 분석 횟수
MAX_PIXELS = 40_000_000        # 이보다 픽셀이 많은 이미지는 거부 (decompression bomb 방지)
API_TIMEOUT = 60
MAX_TOKENS = 1500

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

FEATURE_EMOJI = {
    "이마": "🧠", "눈썹": "〰️", "눈": "👁️", "코": "👃",
    "입": "👄", "귀": "👂", "얼굴형": "🙂",
}
FORTUNE_EMOJI = {"재물운": "💰", "연애운": "💕", "직업운": "💼", "건강운": "🍀"}

DISCLAIMER = "※ 재미로 보는 관상입니다. 과학적 근거가 없으며, 성격·능력·운명을 판단하는 근거로 사용할 수 없습니다."
PRIVACY_NOTE = "🔒 업로드한 사진은 분석에만 사용되며 서버에 저장되지 않습니다."


class AnalysisError(Exception):
    """사용자에게 보여줄 메시지를 담은 분석 오류."""


# ---------------------------------------------------------------- 설정

def _get_setting(name: str) -> str | None:
    """st.secrets → 환경변수(.env) 순서로 설정값을 읽는다."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        # secrets.toml이 없으면 예외가 나므로 .env로 넘어간다.
        pass
    return os.getenv(name) or None


def load_config() -> tuple[str | None, str, int]:
    """API 키, 모델명, 하루 전체 분석 한도를 반환한다."""
    load_dotenv()
    api_key = _get_setting("OPENAI_API_KEY")
    model = _get_setting("OPENAI_MODEL") or DEFAULT_MODEL
    try:
        daily_limit = int(_get_setting("DAILY_LIMIT") or DEFAULT_DAILY_LIMIT)
    except ValueError:
        daily_limit = DEFAULT_DAILY_LIMIT
    return api_key, model, daily_limit


# ---------------------------------------------------------------- 전체 사용량 제한

@st.cache_resource
def _global_usage() -> dict:
    """모든 세션이 공유하는 하루 사용량 카운터 (새로고침해도 초기화되지 않음)."""
    return {"date": None, "count": 0, "lock": threading.Lock()}


def try_reserve_global_call(daily_limit: int) -> bool:
    """오늘 전체 한도가 남아 있으면 1회 차감하고 True를 반환한다."""
    usage = _global_usage()
    today = datetime.date.today()
    with usage["lock"]:
        if usage["date"] != today:
            usage["date"], usage["count"] = today, 0
        if usage["count"] >= daily_limit:
            return False
        usage["count"] += 1
        return True


def release_global_call() -> None:
    """API 호출이 실패하면 차감했던 1회를 되돌린다."""
    usage = _global_usage()
    with usage["lock"]:
        usage["count"] = max(0, usage["count"] - 1)


# ---------------------------------------------------------------- 이미지

def preprocess_image(data: bytes) -> bytes:
    """EXIF 회전 보정, RGB 변환, 긴 변 MAX_SIDE 리사이즈 후 JPEG 바이트로 반환한다."""
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise AnalysisError("사진 해상도가 너무 큽니다. 더 작은 사진을 올려주세요.")
        img = ImageOps.exif_transpose(img)
        img.load()
    except Image.DecompressionBombError as e:
        raise AnalysisError("사진 해상도가 너무 큽니다. 더 작은 사진을 올려주세요.") from e
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        raise AnalysisError("이미지 파일을 읽을 수 없습니다. 손상되지 않은 JPG/PNG/WEBP 사진을 올려주세요.") from e

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # 투명 배경은 흰색으로 채운다.
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    else:
        img = img.convert("RGB")

    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def encode_image(data: bytes, mime_type: str = "image/jpeg") -> str:
    """이미지 바이트를 base64 data URI로 변환한다."""
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


# ---------------------------------------------------------------- 분석

def analyze_face(client: OpenAI, model: str, image_uri: str) -> str:
    """OpenAI Vision 모델에 이미지를 보내 관상 풀이(JSON 문자열)를 받아온다."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_uri, "detail": "high"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=MAX_TOKENS,
        )
    except openai.AuthenticationError as e:
        raise AnalysisError("API 키를 확인해주세요. 키가 올바르지 않거나 만료되었습니다.") from e
    except openai.RateLimitError as e:
        raise AnalysisError("요청이 너무 많거나 사용 한도를 초과했습니다. 잠시 후 다시 시도해주세요.") from e
    except openai.APITimeoutError as e:
        raise AnalysisError("응답 시간이 초과되었습니다. 다시 시도해주세요.") from e
    except openai.APIConnectionError as e:
        raise AnalysisError("OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인하고 다시 시도해주세요.") from e
    except openai.APIError as e:
        raise AnalysisError(f"분석 중 오류가 발생했습니다. 다시 시도해주세요. ({type(e).__name__})") from e

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise AnalysisError("AI가 이 사진의 분석을 거절했습니다. 얼굴이 잘 보이는 다른 사진으로 시도해주세요.")
    content = message.content or ""
    if not content.strip():
        raise AnalysisError("AI가 빈 응답을 보냈습니다. 다시 시도해주세요.")
    return content


def _clamp_score(value) -> int | None:
    """1~5 정수로 맞춘다. 숫자가 아니면 None (화면에는 '–'로 표시)."""
    if isinstance(value, bool):
        return None
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, score))


def _to_bool(value, default: bool = True) -> bool:
    """JSON의 true/false뿐 아니라 "false" 같은 문자열도 올바르게 해석한다."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("false", "no", "0", "아니오", "없음"):
            return False
        if lowered in ("true", "yes", "1", "예", "있음"):
            return True
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _to_keywords(value) -> list[str]:
    """키워드가 리스트가 아니라 "a, b, c" 문자열로 와도 리스트로 만든다."""
    if isinstance(value, str):
        value = value.replace("·", ",").split(",")
    if not isinstance(value, list):
        return []
    return [str(k).strip() for k in value if str(k).strip()][:3]


def _stars(score: int | None) -> str:
    return "–" if score is None else "★" * score + "☆" * (5 - score)


def parse_result(raw: str) -> dict | None:
    """모델 응답을 정규화된 dict로 변환한다. JSON이 아니면 None."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    features = data.get("features") if isinstance(data.get("features"), dict) else {}
    fortune = data.get("fortune") if isinstance(data.get("fortune"), dict) else {}
    face_detected = _to_bool(data.get("face_detected"), default=True)

    normalized_fortune = {}
    for key in FORTUNE_KEYS:
        item = fortune.get(key)
        if not isinstance(item, dict):
            item = {}
        normalized_fortune[key] = {
            "score": _clamp_score(item.get("score")),
            "comment": str(item.get("comment") or ""),
        }

    summary = str(data.get("summary") or "").strip()
    has_features = any(str(features.get(k) or "").strip() for k in FEATURE_KEYS)
    if face_detected and not summary and not has_features:
        # 내용이 없는 응답으로 가짜 결과를 만들지 않고 원문 fallback으로 보낸다.
        return None

    return {
        "face_detected": face_detected,
        "message": str(data.get("message") or ""),
        "summary": summary,
        "features": {k: str(features.get(k) or "").strip() or "판단 어려움" for k in FEATURE_KEYS},
        "fortune": normalized_fortune,
        "keywords": _to_keywords(data.get("keywords")),
        "advice": str(data.get("advice") or ""),
    }


def result_to_text(result: dict) -> str:
    """다운로드용 텍스트를 만든다."""
    lines = ["[AI 관상소 결과]", "", f"첫인상: {result['summary']}", ""]
    lines.append("■ 부위별 관상")
    for key in FEATURE_KEYS:
        lines.append(f"- {key}: {result['features'][key]}")
    lines += ["", "■ 종합 운세"]
    for key in FORTUNE_KEYS:
        item = result["fortune"][key]
        score_text = "–" if item["score"] is None else f"{item['score']}/5"
        lines.append(f"- {key} {_stars(item['score'])} ({score_text}): {item['comment']}")
    lines += ["", f"■ 어울리는 키워드: {', '.join(result['keywords'])}", ""]
    lines += [f"■ 한마디 조언: {result['advice']}", "", DISCLAIMER]
    return "\n".join(lines)


# ---------------------------------------------------------------- 상태

def init_state() -> None:
    st.session_state.setdefault("results", {})      # 이미지 해시 → 결과
    st.session_state.setdefault("call_count", 0)
    st.session_state.setdefault("input_key", 0)     # 위젯 초기화용


def reset_state() -> None:
    st.session_state["results"] = {}
    st.session_state["input_key"] += 1  # 업로더/카메라 위젯을 새로 만든다.


# ---------------------------------------------------------------- 렌더링

def inject_css() -> None:
    """style.css를 페이지에 주입한다."""
    css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _esc(text: str) -> str:
    """모델 응답을 HTML에 넣기 전에 이스케이프한다."""
    return html.escape(str(text)).replace("\n", "<br>")


def render_header() -> None:
    _html(
        '<div class="gs-hero">'
        '<div class="gs-seal">觀<br>相</div>'
        "<div>"
        '<div class="gs-title" role="heading" aria-level="1">AI 관상소</div>'
        '<p class="gs-sub">사진 한 장으로 보는 나의 관상 풀이</p>'
        "</div>"
        "</div>"
    )


def render_notes() -> None:
    _html(
        f'<p class="gs-note">{_esc(DISCLAIMER)}</p>'
        f'<p class="gs-note">{_esc(PRIVACY_NOTE)}</p>'
    )


def render_section(title: str) -> None:
    _html(f'<div class="gs-section">{_esc(title)}</div>')


def render_result(result: dict) -> None:
    pills = "".join(f'<span class="gs-pill">{_esc(k)}</span>' for k in result["keywords"])
    _html(
        '<div class="gs-summary">'
        '<div class="gs-summary-label">첫인상</div>'
        f'<div class="gs-summary-text">{_esc(result["summary"] or "한마디로 정하기 어려운 다채로운 상")}</div>'
        f'<div class="gs-pills">{pills}</div>'
        "</div>"
    )

    render_section("🔍 부위별 관상")
    cards = "".join(
        f'<div class="gs-card" style="animation-delay:{i * 0.05:.2f}s">'
        '<div class="gs-feature-head">'
        f'<span class="gs-feature-icon">{FEATURE_EMOJI[key]}</span>{_esc(key)}'
        "</div>"
        f'<div class="gs-feature-body">{_esc(result["features"][key])}</div>'
        "</div>"
        for i, key in enumerate(FEATURE_KEYS)
    )
    _html(f'<div class="gs-grid">{cards}</div>')

    render_section("🌟 종합 운세")
    rows = ""
    for key in FORTUNE_KEYS:
        item = result["fortune"][key]
        score = item["score"]
        label = "점수 없음" if score is None else f"5점 만점에 {score}점"
        width = 0 if score is None else score * 20
        score_text = "–" if score is None else f"{score}/5"
        rows += (
            '<div class="gs-fortune-row">'
            f'<div class="gs-fortune-name">{FORTUNE_EMOJI[key]} {_esc(key)}</div>'
            '<div class="gs-meter">'
            f'<span class="gs-stars" aria-label="{label}">{_stars(score)}</span>'
            f'<div class="gs-bar"><span style="width:{width}%"></span></div>'
            f'<span class="gs-score">{score_text}</span>'
            "</div>"
            f'<div class="gs-fortune-comment">{_esc(item["comment"])}</div>'
            "</div>"
        )
    _html(f'<div class="gs-card gs-fortune">{rows}</div>')

    render_section("📜 한마디 조언")
    _html(f'<div class="gs-advice">{_esc(result["advice"])}</div>')


def render_empty(message_title: str, message_body: str) -> None:
    _html(
        '<div class="gs-empty">'
        f"<strong>{_esc(message_title)}</strong>{_esc(message_body)}"
        "</div>"
    )


def get_image_input() -> tuple[bytes | None, str]:
    """업로드/카메라 탭에서 이미지 바이트를 받는다."""
    key = st.session_state["input_key"]
    tab_upload, tab_camera = st.tabs(["📁 사진 업로드", "📷 카메라 촬영"])
    with tab_upload:
        uploaded = st.file_uploader(
            "얼굴이 잘 보이는 정면 사진을 올려주세요",
            type=ALLOWED_TYPES,
            key=f"uploader_{key}",
        )
        _html(
            '<p class="gs-hint">JPG · PNG · WEBP 형식, 10MB 이하만 올릴 수 있어요. '
            "빨간 표시가 뜨면 지원하지 않는 형식이거나 용량이 너무 큰 파일입니다.</p>"
        )
    with tab_camera:
        captured = st.camera_input("카메라로 촬영하기", key=f"camera_{key}")

    if uploaded is not None and captured is not None:
        st.info("업로드한 사진과 촬영한 사진이 모두 있어 **업로드한 사진**으로 분석합니다. "
                "촬영한 사진을 쓰려면 업로드한 파일을 지워주세요.")

    source = uploaded or captured
    if source is None:
        return None, ""
    return source.getvalue(), source.name


def request_retry() -> None:
    st.session_state["retry_requested"] = True


def main() -> None:
    st.set_page_config(page_title="AI 관상소", page_icon="🔮", layout="wide")
    inject_css()
    init_state()

    render_header()
    render_notes()

    api_key, model, daily_limit = load_config()
    if not api_key:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다. "
            "로컬에서는 .env 파일에, Streamlit Cloud에서는 Secrets에 OPENAI_API_KEY를 설정해주세요."
        )
        st.stop()

    left, right = st.columns([2, 3], gap="large")

    with left:
        render_section("📷 사진")
        data, name = get_image_input()

    if data is None:
        with right:
            render_empty(
                "사진을 기다리고 있습니다",
                "왼쪽에서 얼굴이 잘 보이는 정면 사진을 올리거나 촬영해주세요.",
            )
        return

    if len(data) > MAX_FILE_BYTES:
        with left:
            st.error("파일이 너무 큽니다. 10MB 이하의 사진을 올려주세요.")
        return

    # 화면에 그리기 전에 먼저 이미지를 검증한다. (손상/초대형 이미지로 앱이 죽지 않게)
    try:
        processed = preprocess_image(data)
    except AnalysisError as e:
        with left:
            st.error(str(e))
            if st.button("다시 하기", width="stretch"):
                reset_state()
                st.rerun()
        return

    image_hash = hashlib.sha256(data).hexdigest()
    results = st.session_state["results"]
    remaining = MAX_CALLS_PER_SESSION - st.session_state["call_count"]

    with left:
        st.image(processed, caption="분석할 사진", width="stretch")
        col1, col2 = st.columns(2)
        analyze_clicked = col1.button("관상 보기", type="primary", width="stretch")
        if col2.button("다시 하기", width="stretch"):
            reset_state()
            st.rerun()

    retry_requested = st.session_state.pop("retry_requested", False)

    with right:
        if (analyze_clicked or retry_requested) and image_hash not in results:
            if remaining <= 0:
                st.warning(f"이 세션에서 분석할 수 있는 횟수({MAX_CALLS_PER_SESSION}회)를 모두 사용했습니다.")
                return
            if not try_reserve_global_call(daily_limit):
                st.warning("오늘 준비된 분석 횟수가 모두 소진되었습니다. 내일 다시 찾아주세요.")
                return
            try:
                client = OpenAI(api_key=api_key, timeout=API_TIMEOUT, max_retries=1)
                with st.spinner("관상을 살피는 중..."):
                    raw = analyze_face(client, model, encode_image(processed))
            except AnalysisError as e:
                release_global_call()
                st.error(str(e))
                st.button("🔄 다시 시도", on_click=request_retry, width="stretch")
                return
            st.session_state["call_count"] += 1
            results[image_hash] = {"raw": raw, "parsed": parse_result(raw)}
            remaining -= 1

        entry = results.get(image_hash)
        if entry is None:
            render_empty(
                "준비되었습니다",
                f"'관상 보기' 버튼을 눌러주세요. (남은 분석 횟수: {max(remaining, 0)}회)",
            )
            return

        parsed = entry["parsed"]
        if parsed is None:
            st.warning("결과를 정리된 형식으로 읽지 못해 원문을 그대로 보여드립니다.")
            st.text(entry["raw"])  # markdown으로 렌더링하지 않음 (외부 이미지/링크 차단)
            download_text = entry["raw"]
        elif not parsed["face_detected"]:
            guide = "얼굴이 잘 보이는 한 사람의 정면 사진을 올려주세요."
            reason = parsed["message"].strip()
            st.warning(reason if "정면 사진" in reason else f"{reason} {guide}".strip())
            return
        else:
            render_result(parsed)
            download_text = result_to_text(parsed)

        _html('<div class="gs-footer"></div>')
        st.download_button(
            "📥 결과 다운로드 (.txt)",
            data=download_text.encode("utf-8"),
            file_name="관상_결과.txt",
            mime="text/plain",
            width="stretch",
        )
        render_notes()


if __name__ == "__main__":
    main()
