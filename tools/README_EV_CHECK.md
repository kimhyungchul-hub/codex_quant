# EV 값 확인 및 음수 EV 원인 분석 가이드

## 현재 상태

실행 중인 프로세스 (PID: 77606)가 포트 9999에서 Dashboard를 제공하고 있습니다.

## EV 값 확인 방법

### 방법 1: Dashboard WebSocket (권장)

1. 브라우저에서 `http://localhost:9999` 접속
2. 개발자 도구 열기 (F12)
3. Network 탭 > WS (WebSocket) 선택
4. WebSocket 메시지에서 payload 확인
5. `ev` 필드 값 확인

### 방법 2: Payload JSON 파일로 분석

1. Dashboard에서 payload를 JSON으로 저장
2. 저장한 파일로 분석:
   ```bash
   python3 check_negative_ev.py <payload.json>
   ```

### 방법 3: 로그 확인

실행 중인 프로세스의 stdout/stderr에서 `[EV_DEBUG]` 태그 확인:
```bash
# 프로세스의 출력 확인 (터미널에서)
# 또는 로그 파일 확인
tail -f state/engine_run.log | grep "\[EV_DEBUG\]"
```

## 음수 EV 원인 분석

EV가 음수이거나 작은 양수(< 0.0005)인 경우, `check_negative_ev.py`가 4개 항목을 자동으로 점검합니다:

1. **mu_alpha 자체가 음수/미약**
   - `mu_alpha`, `mu_alpha_mom`, `mu_alpha_ofi` 확인

2. **exit이 min_hold 근처에서 반복적으로 발생**
   - `policy_exit_time_mean_sec` ≈ 180s 확인

3. **maker delay + spread + slippage가 gross EV를 잠식**
   - `pmaker_entry_delay_sec`, `fee_roundtrip_total` 확인
   - `ev_decomp_gross_long_600` vs `ev_decomp_net_long_600` 비교

4. **SHORT 쪽 EV가 상대적으로 더 좋은데 long 편향**
   - `policy_ev_gap`, `policy_p_pos_gap` 확인
   - `policy_ev_mix_long` vs `policy_ev_mix_short` 비교

## 사용 가능한 스크립트

- `check_negative_ev.py`: 음수 EV 원인 분석
- `extract_ev_from_runtime.py`: 런타임에서 EV 값 추출 시도
- `fetch_and_check_ev.py`: Dashboard에서 payload 가져오기 시도
- `run_ev_verification.sh`: EV 검증용 런타임 실행


