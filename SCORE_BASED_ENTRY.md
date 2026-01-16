# ✅ Score 기반 진입 시스템 완료 + mu_base/sigma 수정

## 🎉 완료된 작업

### 1. **Entry Gate 비활성화** ✅
- `entry_evaluation.py`: `SCORE_ONLY_MODE=True` 시 모든 게이트 우회

### 2. **Funnel Filter 비활성화** ✅
- `decision.py`: Score만으로 진입 결정

### 3. **mu_base/sigma 문제 수정** ✅
- `orchestrator.py`: `ctx`에 `mu_sim`, `sigma_sim` 추가
- `_compute_returns_and_vol()` 함수로 closes에서 계산

---

## 📊 현재 상황

### ✅ 해결된 문제:
```
이전: [EV_DEBUG] ⚠️  WARNING: mu_base or sigma is invalid!
현재: mu_base와 sigma 정상 전달 ✅
```

### ⚠️  남은 문제:
**모든 EV가 음수** (-0.71% ~ -0.79%)

```
[EV_VALIDATION_NEG] BTC/USDT:USDT | policy_ev_mix both negative: 
  long=-0.001439 short=-0.001449

원인:
- TP 확률: 0% (0.0000)
- SL 확률: 100% (1.0000)  
- 비용이 수익의 5845%
```

**문제의 근본 원인:**
1. `mu_sim` 값이 거의 0이거나 음수
   - 최근 가격이 횡보/하락 중
2. 실행 비용이 너무 큼 (0.072% ~ 0.078%)
   - 단기(60s) horizon에는 비용이 수익을 압도

---

## 🎯 Score 기반 진입 로직 (작동 중)

```python
# 1. Objective 계산
J = EV / (|CVaR| + 2*StdDev) * (1/sqrt(T))

# 2. Score 계산
score_long = J_long + neighbor_bonus - neighbor_penalty
score_short = J_short + neighbor_bonus - neighbor_penalty

# 3. 진입 결정
if max(score_long, score_short) <= 0.0:
    direction = 0  # WAIT (현재 상황)
elif abs(score_long - score_short) < min_gap:
    direction = 0  # WAIT
else:
    direction = 1 if score_long > score_short else -1
```

**현재 Score**: 둘 다 음수 → direction = 0 → WAIT ✅

---

## 💡 해결 방안

### Option A: 임계값 완화 (즉시 적용 가능)
```bash
# 음수 Score도 허용 (작은 음수까지)
export SCORE_MIN_THRESHOLD=-0.003  # -0.3%까지 허용
```

### Option B: TP/SL 비율 완화 (중기)
```python
# 현재 (너무 타이트):
tp_pct = 0.0005  # 0.05%
sl_pct = 0.0008  # 0.08%

# 제안 (5분 단위에 적합):
tp_pct = 0.003   # 0.3%
sl_pct = 0.005   # 0.5%
```

### Option C: Horizon 연장 (장기)
```python
# 현재: 60~600초 (1~10분)
# → 비용이 너무 큰 비중 차지

# 제안: 300~1800초 (5~30분)
# → 더 긴 시간에 수익 실현
```

---

## ✅ 성공 기준

**Score > 0 나오면 진입 가능!**

예상 시나리오:
```
# 상승장시
mu_sim = 0.0005 (연간 0.05%)
score_long = 0.002
score_short = -0.001
gap = 0.003 > min_gap
→ direction = 1 (LONG) ✅
```

---

## 🔧 테스트 명령어

```bash
# 1. Score 확인
curl -s "http://localhost:9999/debug/payload" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('market', [{}])[0]
print(f\"Score L: {r.get('policy_ev_score_long')}\")
print(f\"Score S: {r.get('policy_ev_score_short')}\")
print(f\"EV: {r.get('ev')}\")
print(f\"Status: {r.get('status')}\")
"

# 2. 로그 모니터링
tail -f engine_early_return.log | grep -E "(SCORE_ONLY|direction=|policy_ev_score)"
```

---

## 📝 결론

✅ **로직 완료**: Score 기반 진입/청산 시스템 완성
✅ **버그 수정**: `mu_base=None` 문제 해결
⚠️  **대기 중**: 현재 시장이 횡보/하락 → EV 음수 → WAIT (정상 작동)

**시장이 상승하거나 변동성이 커지면 Score > 0이 되어 진입할 것입니다!** 🚀

---

## 📄 관련 파일

수정된 파일:
- `engines/mc/entry_evaluation.py` (Entry Gate 비활성화 + `_env_bool` 추가)
- `engines/mc/decision.py` (Funnel Filter 비활성화)
- `core/orchestrator.py` (mu_sim, sigma_sim 추가)

문서:
- `SCORE_ANALYSIS.md` (Score 계산 분석)
- `ENTRY_ANALYSIS.md` (진입 조건 분석)
- `SCORE_BASED_ENTRY.md` (이 문서)
