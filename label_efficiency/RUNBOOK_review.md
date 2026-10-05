# 재학습 + 리뷰 대응 분석 런북

Reviewer 2의 두 지적(C3 동일 오검출 조건 비교, C4 세션별 결과)에 답하기 위한 절차입니다.
**로봇·Coral은 필요 없습니다. GPU 머신만 있으면 됩니다.**

이 문서는 PC에서 Claude CLI에 그대로 넘겨도 되고, 직접 실행해도 됩니다.

---

## 왜 재학습이 필요한가

저장된 36개 런의 JSON에는 집계값과 클래스별 값만 있습니다.

```
results.test = { overall: {...}, per_class: {...} }
```

세션별 분해도, 이미지별 점수도 없습니다. 그리고 `de_train.py`의 모델 저장은
`--save-model`이 있을 때만 동작하므로 **36개 런의 가중치가 남아 있지 않습니다.**
두 분석 모두 모델의 예측이 필요하므로 모델을 남기는 재학습이 선행돼야 합니다.

필요한 런은 36개가 아니라 **6개**입니다 — 전체 예산(fraction 100)에서
heatmap과 softargmax를 seed 0/1/2로. 비교 대상이 이 두 head이고,
라벨 예산 곡선 자체는 이미 Table 3에 있습니다.

| 단계 | 작업 | 소요 |
|---|---|---|
| 1 | 파일 배치 + 사전 점검 | 5분 |
| 2 | 6런 재학습 (`--save-model`) | 약 2.6시간 |
| 3 | 모델별 분석 (추론만) | 10–20분 |
| 4 | 표 생성 | 1분 |

---

## 0. 전제

`de_train.py`는 **수정하지 않습니다.** `--save-model`이 이미 있습니다
(원래 `quantize_eval.py`용으로 추가한 옵션).

```
ap.add_argument('--save-model', default=None, ...)
...
if a.save_model:
    model.save(a.save_model)
```

새로 추가하는 것은 분석 스크립트 2개와 실행 스크립트 1개뿐입니다.

---

## 1. 파일 배치와 사전 점검

세 파일을 `label_efficiency/` 안에 넣습니다 (`de_train.py`, `de_common.py`,
`splits.json`과 같은 디렉터리).

```
label_efficiency/
├── de_train.py          기존, 수정 없음
├── de_common.py         기존
├── splits.json          기존
├── review_eval.py       ← 추가
├── review_tables.py     ← 추가
└── retrain_review.sh    ← 추가
```

점검:

```bash
cd label_efficiency
python3 -c "import de_common as C, de_train as T; print(C.INPUT_SIZE, C.N_CLASSES)"
python3 -c "
import de_common as C, json
sp = C.load_splits('splits.json')
print('train pool', len(sp['subsets']['100']), 'val', len(sp['val']), 'test', len(sp['test']))
print('test_sessions', sp.get('test_sessions'))
print('sample path', sp['test'][0])
"
```

**기대값:** `1193 / 211 / 492`, `test_sessions ['Dataset4', 'Dataset1_Jeehyun']`.

`sample path`를 꼭 확인하세요. 경로에 세션 이름이 디렉터리로 들어 있어야
세션별 분해가 됩니다. 예: `.../Localization Dataset/Dataset4/0001.jpg`.
들어 있지 않으면 3단계에서 `unassigned` 경고가 나오고, 그때
`review_eval.py`의 `session_of()`를 그 경로 구조에 맞춰 고치면 됩니다.

> **지난번 걸렸던 함정:** `splits.json`의 경로가 상대 경로입니다.
> `can't open/read file` 뒤에 `need at least one array to stack`이 나오면
> 데이터셋이 작업 디렉터리에서 보이지 않는 것입니다.
> `splits.json` 옆에 데이터셋 심볼릭 링크를 걸는 것이 가장 안전한 해결입니다.

---

## 2. 재학습 (약 2.6시간)

**먼저 1런만 돌려 설정을 확인합니다.** 2.6시간을 날리지 않기 위한 단계입니다.

```bash
bash retrain_review.sh heatmap 0
```

약 26분. 끝나면 두 파일이 생깁니다.

```
models/heatmap_f100_s0.keras        ← 새로 남기는 가중치
runs_review/heatmap_f100_s0.json    ← 기존 형식과 동일한 런 요약
```

`runs_review/heatmap_f100_s0.json`의 `results.test.overall.pck10`이
논문 Table 3의 heatmap 전체 예산 값과 **대략** 맞는지 보세요.
GPU 비결정성 때문에 자리까지 같지는 않습니다 (3단계 주의사항 참조).

여기까지 정상이면 **3단계를 이 모델 하나로 먼저 돌려 분석 스크립트를 검증한 뒤**,
남은 5런을 돌립니다.

```bash
bash retrain_review.sh            # 완료된 런은 자동으로 건너뜀
```

중간에 끊겨도 다시 실행하면 됩니다. 출력은 `runs_review/`, `models/`,
`logs_review/`로 나가므로 **기존 `runs/`의 36개 런은 건드리지 않습니다.**

백그라운드로 돌릴 경우:

```bash
nohup bash retrain_review.sh > logs_review/all.log 2>&1 &
tail -f logs_review/all.log
```

---

## 3. 모델별 분석 (추론만, 10–20분)

```bash
mkdir -p results/review
for head in heatmap softargmax; do
  for seed in 0 1 2; do
    tag="${head}_f100_s${seed}"
    [ -f "models/$tag.keras" ] || continue
    echo "== $tag"
    python3 review_eval.py \
      --splits splits.json \
      --model "models/$tag.keras" \
      --head "$head" \
      --out "results/review/$tag.json"
  done
done
```

`review_eval.py`가 하는 일:

1. val 211장과 test 492장을 한 번만 forward (비싼 단계)
2. 디코딩 임계값을 0.05–0.95로 훑으며 매 지점의 recall / PCK / **이미지당 오검출 수** 기록
3. test를 세션별로 쪼개 같은 지표를 각 세션에 대해 기록

임계값의 의미는 head마다 다릅니다 — 그게 핵심입니다.

| head | 훑는 값 | 배포값 |
|---|---|---|
| heatmap | `decode_heatmap`의 peak 임계값 (`min_score`는 1.0 고정) | 0.30 |
| softargmax | visibility logit의 sigmoid 컷오프 | 0.50 |

숫자로는 비교할 수 없고, **각 값이 만들어내는 이미지당 오검출 수를 통해서만**
비교됩니다. 4단계가 그 축을 맞춥니다.

출력 예:

```
  test thr 0.30  recall 0.903  PCK@10 0.436  fp/img 0.331  <- deployed
  session Dataset4             n= 287  recall 0.912  PCK@10 0.452  mean 15.88 px
  session Dataset1_Jeehyun     n= 205  recall 0.890  PCK@10 0.414  mean 17.21 px
```

> **주의 — 논문 Table 3과 섞지 마세요.** 이 6런은 새 체크포인트입니다.
> 같은 seed라도 GPU 연산 순서 때문에 비트 단위로 재현되지 않습니다.
> 새 표로 따로 제시하고, 캡션에 "리뷰 대응을 위해 동일 설정으로 재학습한
> 별도 런"이라고 명시해야 합니다. quantization 표에서 겪은 것과 같은 문제입니다.

---

## 4. 표 생성

```bash
python3 review_tables.py results/review/*.json --out results/review
```

생성물:

| 파일 | 대응하는 지적 |
|---|---|
| `results/review/table_sessions.tex` | R2-C4 — 세션별 held-out 결과, seed 3개 평균 ± 표본 SD |
| `results/review/table_matched.tex` | R2-C3 — 오검출 수를 맞춘 head 비교 |
| `results/review/review_summary.json` | 위 두 표의 근거 수치 |

**matched 표의 프로토콜** (여기가 틀리기 쉬운 부분입니다):

- 목표 오검출 수를 두 개 잡습니다 — 각 head의 배포 설정에서 나온 검증 오검출 수.
  논문 기준으로 heatmap 약 0.33/장, softargmax 약 0.15/장. 즉 **양방향으로**
  비교합니다. heatmap을 조여 softargmax에 맞추고, softargmax를 풀어 heatmap에 맞춥니다.
  한쪽 방향만 하면 "유리한 쪽으로만 맞췄다"는 재지적을 받습니다.
- 임계값은 **검증 split에서** 목표에 가장 가까운 지점으로 고르고,
  보고하는 숫자는 **held-out**의 같은 임계값 결과입니다.
  held-out에서 고르면 19번 시도 중 최선을 한 번의 측정처럼 보고하는 셈이 됩니다.
- 산포는 표본 SD(ddof=1)로, 논문 Table 3과 같은 규약입니다.

목표값을 직접 지정하려면:

```bash
python3 review_tables.py results/review/*.json --out results/review --targets 0.15,0.33
```

### `NOT MATCHED` 경고가 나오면

한쪽 head가 임계값 격자 안에서 목표 오검출 수에 **도달하지 못하는** 경우가 있습니다.
softargmax를 아무리 느슨하게 풀어도 heatmap만큼 오검출이 많이 나오지 않는 상황입니다.
이때 스크립트는 경고를 출력하고 해당 행에 `$^{\dagger}$`를 붙입니다.

```
target 0.405  softargmax  reached only 0.230   <- NOT MATCHED
```

**이 행은 "맞춘 비교"로 제시하면 안 됩니다.** 그 목표를 빼거나,
"softargmax는 임계값을 최대로 풀어도 이미지당 0.23건을 넘지 않는다"는
도달 가능 범위로 서술하는 쪽이 맞습니다. 후자가 오히려 R2에게는 더 강한 답변입니다 —
두 head의 동작점 범위 자체가 겹치지 않는다는 뜻이므로.

반대 방향(heatmap을 조여 softargmax에 맞추는 쪽)은 거의 항상 가능합니다.
그 행 하나만으로도 R2-C3에 답할 수 있습니다.

---

## 5. 보내줄 것

다음을 보내주시면 원고에 들어갈 표·본문·응답 서신 문구를 작성합니다.

```
results/review/table_sessions.tex
results/review/table_matched.tex
results/review/review_summary.json
results/review/*.json          (6개, 용량이 크면 생략 가능)
runs_review/*.json             (6개)
```

`review_eval.py`의 콘솔 출력도 같이 주시면 좋습니다 — 세션 매칭이
제대로 됐는지 확인할 수 있습니다.

---

## 선택 사항: leave-one-session-out

R2-C4의 더 강한 형태입니다. 위 4단계까지로도 "세션별 결과"는 제시되므로
**필수는 아닙니다.** 하기로 하면 세션 분할을 새로 만들어야 하고
(`splits.json`을 fold별로 생성), 세션 수 × 1 head × 1 seed 만큼 학습이 듭니다.
세션이 5개면 약 2.2시간/head. 이건 위 결과를 받은 뒤에 판단하는 게 맞습니다.

---

## 실패 시 확인 순서

| 증상 | 원인 |
|---|---|
| `need at least one array to stack` | `splits.json`의 상대 경로. 데이터셋 심볼릭 링크 |
| `model.save` 확장자 오류 | Keras 3는 `.keras` 확장자 필요. 스크립트는 이미 `.keras` 사용 |
| `Unknown layer: SoftArgmax2D` | `review_eval.py`를 `de_train.py`와 같은 디렉터리에서 실행할 것 |
| 세션별 출력에 `unassigned` | 경로에 세션 이름이 없음. `session_of()` 수정 |
| OOM | `--batch 16` (분석) / `de_train.py --batch 8` (학습) |
| 재학습 PCK가 Table 3과 많이 다름 | `--fraction 100 --lr 1e-3 --cosine` 기본값이 맞는지 로그 확인 |
