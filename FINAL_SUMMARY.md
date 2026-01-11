# ✅ 완료: EV 숨김 + Score 임계값 완화

## 🎯 완료된 작업

### 1. **대시보드에서 EV 숨김** ✅
- `dashboard_v2.html` 수정
- 테이블 헤더 및 셀에 `display:none` 적용
- Score만 표시됨

### 2. **Score 임계값 완화** ✅
- 기존: `score > 0.0` (너무 타이트)
- **현재: `score > -0.01`** (1% 손실까지 허용)
- 환경변수 `SCORE_ENTRY_THRESHOLD`로 조정 가능

---

## 📊 Score 임계값 상세

### 기본 설정
```bash
export SCORE_ENTRY_THRESHOLD=-0.01  # -1%까지 허용
```

### 다른 옵션
```bash
# 보수적
export SCORE_ENTRY_THRESHOLD=-0.005  # -0.5%까지

# 중간 (기본)
export SCORE_ENTRY_THRESHOLD=-0.01   # -1%까지

# 공격적
export SCORE_ENTRY_THRESHOLD=-0.02   # -2%까지

# 매우 보수적
export SCORE_ENTRY_THRESHOLD=0.0     # 양수만 (원래 설정)
```

---

## 🎨 대시보드 변경사항

### Before:
```
| SYM | PRICE | STATUS | EV% | SCORE% | ...
| BTC | 42000 | WAIT   | -0.73 | -0.031 | ...
```

### After:
```
| SYM | PRICE | STATUS | SCORE% | ...
| BTC | 42000 | WAIT   | -0.031 | ...
```

**EV 컬럼 완전 숨김** ✅

---

## 📈 예상 동작

### 현재 상황 (Score ≈ -0.03)
```
임계값: -0.01
score_long = -0.031 < -0.01  ❌ 진입 불가
score_short = -0.033 < -0.01 ❌ 진입 불가
→ WAIT
```

### 시장이 약간 움직이면
```
score_long = -0.008 > -0.01  ✅ 진입 가능!
score_short = -0.033 < -0.01 ❌
→ LONG 진입!
```

### 하락장
```
score_long = -0.033 < -0.01  ❌
score_short = -0.005 > -0.01  ✅ 진입 가능!
→ SHORT 진입!
```

---

## 🔧 모니터링

### 1. 대시보드
```
http://localhost:9999
```
- SCORE% 컬럼 확인
- EV는 보이지 않음

### 2. 로그
```bash
tail -f engine_early_return.log | grep -E "(long_only_positive|short_only_positive|threshold)"
```

### 3. API
```bash
curl -s "http://localhost:9999/debug/payload" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('market', [{}])[0]
print(f\"Symbol: {r.get('symbol')}\")
print(f\"Score L: {r.get('policy_ev_score_long')}\")
print(f\"Score S: {r.get('policy_ev_score_short')}\")
print(f\"Status: {r.get('status')}\")
"
```

---

## 📝 수정된 파일

1. **dashboard_v2.html**
   - 라인 390: EV 헤더 숨김
   - 라인 1056: EV 셀 숨김

2. **engines/mc/entry_evaluation.py**
   - 라인 1887-1913: Score 임계값 환경변수 지원

---

## ✅ 결론

**완료**:
1. ✅ EV 대시보드에서 숨김
2. ✅ Score 임계값 -0.01로 완화
3. ✅ 환경변수로 조정 가능

**효과**:
- 시장이 조금만 움직여도 진입 가능
- 롱/숏 독립 판단
- Score 기반 일관된 기준

**다음 모니터링**:
- Score 값 확인
- 진입 발생 여부
- 필요시 임계값 재조정
