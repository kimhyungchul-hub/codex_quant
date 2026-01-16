# Copilot instructions (codex_quant)

## 🤖 Copilot 운영 규칙 (CRITICAL)
1.  **진실의 원천 (Source of Truth):** 너는 이 프로젝트의 시니어 개발자처럼 행동해야 한다. 코드를 생성하기 전에 반드시 아래의 **[전체 구조(Big picture)]**와 **[변경 로그(Change Log)]**를 검토하여 일관성을 유지하라.
2.  **아키텍처 보존:** '연산 루프(Compute)'와 'UI 루프(Refresh)'가 분리된 현재의 구조를 깨뜨리는 코드를 절대 제안하지 마라.
3.  **문서화 루틴:**
    * 새로운 기능을 구현하거나 리팩토링할 때, **[변경 로그]**를 확인하여 코드의 발전 흐름을 파악하라.
    * **답변의 맨 마지막**에는 반드시 **[변경 로그]**에 추가할 한 줄 요약을 제공하고
    * 형식: `[YYYY-MM-DD] 변경 내용 요약 (수정된 파일명)`

---

## 🏗 전체 구조 (Big Picture)
- **진입점 (Entrypoint):** `main.py`. 여기서 `LiveOrchestrator`와 `DashboardServer`를 시작함 (`--no-dashboard` 옵션이 없을 경우).
- **핵심 데이터 흐름:**
  `DataManager` (티커/OHLCV/호가창) → 오케스트레이터가 심볼별 `ctx` 생성 → `EngineHub.decide_batch()` → `_decision_cache` 업데이트 → `_rows_snapshot_cached()` / `_row()` → `_build_payload()`를 통해 대시보드로 전송.
- **오케스트레이터 루프 분리 (의도적 설계):**
  - **무거운 연산:** `LiveOrchestrator.decision_worker_loop()` - `_decision_cache` 업데이트 담당.
  - **UI 갱신:** `LiveOrchestrator.decision_loop()` - `_last_rows` 빌드 및 브로드캐스팅 담당.

## 📂 우선 확인해야 할 핵심 모듈
- **런타임 + 트레이딩 루프:** `core/orchestrator.py`
- **시장 데이터:** `core/data_manager.py`
- **대시보드 페이로드/API:** `core/dashboard_server.py` 및 UI `dashboard_v2.html`
- **MC(몬테카를로) 엔진:** `engines/mc/monte_carlo_engine.py` (믹스인), `engines/mc/entry_evaluation.py`, `engines/mc/exit_policy.py`, `engines/mc/decision.py`
- **사이징/리스크:** `engines/mc/decision.py` (심볼별 비중/레버리지), `core/risk_manager.py` (실시간 안전장치), `core/paper_broker.py` (모의 체결)

## 📏 프로젝트별 컨벤션
- **대시보드 컬럼:** 대시보드 테이블 컬럼은 `dashboard_v2.html`의 `data-k` 키값(예: `evph_p`, `hold_mean_sec`, `rank`, `ev_score_p`, `napv_p`, `evp`, `optimal_horizon_sec`)에 의해 구동됨. 컬럼을 추가/수정할 때는 `LiveOrchestrator._row()` / `_rows_snapshot_cached()`를 업데이트하여 해당 값이 **행(row)의 최상위 레벨**에 존재하도록 해야 함.
- **설정(Config):** `config.py`는 모듈 임포트 시점에 로드됨. `.env`를 먼저 로드하고, `state/bybit.env`가 존재하면 덮어씀. 이 로딩 순서를 고려하지 않고 기본값을 함부로 변경하지 말 것.
- **호환성 유지:** `engines/mc/` 하위에 하위 호환성을 위한 래퍼/재수출(re-exports)이 존재함 (`ARCHITECTURE.md` 참조). 리팩토링 시 임포트 안정성을 유지할 것.
- **JAX 사용:** JAX는 "최선을 다해(best-effort)" 지원됨. `main.py`에서 `JAX_COMPILATION_CACHE_DIR`를 기본 설정하지만, 일부 백엔드(Apple METAL 등)는 영구 컴파일 캐시를 사용하지 않을 수 있음.

## 🛠 개발자 워크플로우 (리포지토리 문서/코드 검증됨)
- **가상환경 생성 및 설치:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
## Change Log
- 2026-01-17: 환경변수 기본값을 config.py로 중앙집중화하고, 레포지토리 내 주요 스크립트/엔진에서 `os.environ.get` 직접 사용을 제거하여 설정 소스를 통일함 (config.py, core/orchestrator.py, core/risk_manager.py, engines/mc/decision.py, scripts/).
