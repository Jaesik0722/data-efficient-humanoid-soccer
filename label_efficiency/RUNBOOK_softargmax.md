# Soft-argmax 공정성 실험 — 실행 안내

리뷰의 핵심 지적 하나에 답하기 위한 실험입니다. 현재 원고 §5.1은 soft-argmax가
learning-rate probe를 받지 않았고 coordinate-loss weight와 softmax temperature도
탐색되지 않았다고 밝히고 있습니다. 이 실험은 그 문장을 사실로 바꿉니다.

**두 단계로 나뉘고, 1단계만으로도 지적의 대부분이 해소됩니다.**

| | 내용 | 비용 | 얻는 것 |
|---|---|---|---|
| 1단계 | soft-argmax에 동일한 lr probe | **약 10분** | "세 방식 모두 같은 probe를 거쳤다" |
| 2단계 | τ × coordinate weight 그리드 | 약 10~11시간 | "탐색해도 결론이 바뀌지 않는다" |

---

## 준비 — 어느 기계에서 돌리나

`runs/`의 36개 학습을 돌린 기계입니다. 데이터셋(`Localization Dataset`)과 GPU가
거기 있습니다. 환경은 `environment.yml` 그대로입니다.

```bash
conda activate robinion-de          # TF 2.15.1 / numpy 1.26.4
cd label_efficiency
python check_env.py --dataset "/path/to/Localization Dataset"
```

`check_env.py`가 GPU를 못 찾으면 먼저 그것부터 해결하세요. CPU로는 1단계도
몇 시간이 걸립니다.

### splits.json은 반드시 원본과 같아야 합니다

새 실행이 기존 36회와 **같은 train/val/test 분할**을 써야 비교가 성립합니다.

- 원본 `splits.json`이 남아 있으면 그것을 쓰세요. 가장 안전합니다.
- 없으면 다시 만들되(`seed=0`이 기본이라 데이터셋이 그대로면 결과가 동일합니다),
  아래 숫자와 일치하는지 **반드시 확인**하고 넘어가세요.

```bash
python3 de_common.py --dataset "/path/to/Localization Dataset" --out splits.json
python3 -c "
import json; s=json.load(open('splits.json'))
print('subsets', {k: len(v) for k,v in s['subsets'].items()})
print('val', len(s['val']), 'test', len(s['test']))
print('test sessions', s['test_sessions'])
"
```

나와야 하는 값 — `runs/heatmap_f100_s0.json`에 기록된 것과 같습니다:

```
subsets {'10': 120, '25': 298, '50': 597, '100': 1193}
val 211  test 492
test sessions ['Dataset4', 'Dataset1_Jeehyun']
```

하나라도 다르면 데이터셋 폴더가 그때와 달라진 것입니다. 그 상태로 돌리면
새 결과를 기존 36회와 같은 표에 놓을 수 없습니다.

---

## 1단계 — lr probe (약 10분)

```bash
chmod +x tune_softargmax.sh
./tune_softargmax.sh splits.json
```

heatmap·coord가 받았던 것과 **완전히 같은 프로토콜**입니다: 3e-4 / 1e-3 / 3e-3,
800 steps, 100% 라벨, seed 0. 결과는 `lr_probe_softargmax.json`.

화면에 rate별로 `recall`, `PCK@10`, `mean px`와 판정(`learning` / `weak` /
`collapsed` / `not learning`)이 찍힙니다.

**결과 읽는 법**

- **1e-3이 최선이면** — 가장 가능성이 높습니다. 원고는 "세 방식 모두 같은 probe를
  거쳐 같은 rate를 선택했다"로 고쳐 쓰면 되고, 비대칭은 heatmap의 목적함수 하나만
  남습니다. 2단계는 선택 사항이 됩니다.
- **다른 rate가 더 좋으면** — soft-argmax가 불리한 rate로 돌았다는 뜻이므로
  그 rate로 12회(4 budget × 3 seed)를 다시 돌려 Table 2를 갱신해야 합니다.
  약 5시간입니다. 이 경우 2단계도 하는 편이 좋습니다.

`lr_probe_softargmax.json`을 보내주시면 곡선을 읽고 어느 쪽인지 판단해 드리겠습니다.

---

## 2단계 — τ × coordinate weight 그리드 (선택, 하룻밤)

### 왜 필요한가

`SoftArgmax2D`에는 **원래 temperature 파라미터가 없었습니다.** 공간 softmax가
`tf.nn.softmax(flat)`, 즉 τ=1 고정이었습니다. 이번 수정으로 `--tau`가 생겼고,
τ=1은 기존 동작과 비트 단위로 동일함을 확인했습니다(아래 "검증" 참조).
coordinate weight도 `--coord-weight`로 노출했으며 기본값 5는 그대로입니다.

### 실행

```bash
chmod +x grid_softargmax.sh
nohup ./grid_softargmax.sh splits.json > gridA.log 2>&1 &
tail -f gridA.log
```

A단계: τ ∈ {0.5, 1, 2} × weight ∈ {1, 5, 20}, 라벨 25%·100%, seed 0 — 18회, 약 8시간.
(τ=1·weight=5 조합은 기존 `runs/softargmax_f25_s0.json`, `..._f100_s0.json`이
이미 있으므로 건너뜁니다. 실제로는 16회입니다.)

끝나면:

```bash
python3 read_grid.py --runs runs
```

**검증셋 PCK@10**으로만 고릅니다 — test 세션은 선택에 관여하지 않습니다.
스크립트가 마지막에 B단계 명령을 그대로 찍어줍니다:

```bash
WINNER="0.5 20" ./grid_softargmax.sh splits.json      # 예시
```

B단계: 승자 설정을 25%·100% × seed 3개로 확인 — 6회, 약 2.5시간.

### 결과 파일 이름

τ나 weight가 기본값이 아니면 파일명에 붙습니다:
`softargmax_f25_s0_t0.5_w20.json`. 따라서 **기존 36개 파일은 절대 덮어쓰이지
않고**, 중단된 스윕을 다시 실행하면 남은 것부터 이어서 돌립니다.

### 어느 결과든 원고는 강해집니다

- **튜닝해도 heatmap에 못 미치면** — 주장 범위를 "명시된 프로토콜에서"로 좁혀 둘
  이유가 없어집니다. 지금 원고에서 가장 방어적인 문단을 지울 수 있습니다.
- **따라잡으면** — 게재 후가 아니라 지금 아는 편이 낫습니다. 그때는 결론을
  "출력 표현의 차이"에서 "기본 설정에서의 견고함(robustness to tuning)"으로
  옮기면 되고, 그것도 실용적으로 의미 있는 주장입니다.

---

## 코드 변경 내역

`de_train.py`

- `SoftArgmax2D.__init__(tau=1.0)` 추가, `softmax(flat / tau)`. 시각화·직렬화를
  위해 `get_config()`도 추가했습니다. visibility 분기는 **의도적으로** 원래
  로짓을 그대로 씁니다 — 학습되는 affine이 이미 스케일 역할을 하므로 τ에 묶으면
  좌표 디코딩과 검출 임계가 섞입니다.
- `coord_loss` / `softargmax_loss`를 `coord_loss_fn(w)` / `softargmax_loss_fn(w)`
  팩토리로 바꾸고, 옛 이름은 `w=5`로 고정한 별칭으로 남겼습니다
  (`tune_lr.py`의 기존 호출이 그대로 동작합니다).
- CLI에 `--tau`, `--coord-weight` 추가. 둘 다 JSON 결과에 기록됩니다.
- 기본값이 아니면 tag에 접미사를 붙여 파일이 충돌하지 않게 했습니다.

`tune_lr.py`

- `--head`에 `softargmax` 추가, 모델·손실 dispatch를 세 갈래로.
- `--tau`, `--coord-weight` 전달.
- heatmap이 아닌 head에서는 all-zeros baseline이 해당 목표에 대한 값이 아니라는
  안내 한 줄을 출력합니다(판정은 원래부터 정확도 기준입니다).

**기본값은 하나도 바뀌지 않았습니다.** τ=1, weight=5, lr=1e-3으로 돌리면 기존
36회와 동일한 설정입니다.

## 검증 (합성 텐서, 데이터셋 불필요)

τ 구현을 다섯 가지로 확인했습니다.

1. τ=1의 출력이 수정 전 코드와 **비트 단위로 동일** — 기존 결과가 그대로 재현됩니다
2. τ→0에서 arg-max 셀 중심으로 수렴 (오차 2.6e-08)
3. τ→∞에서 격자 중심 (0.5, 0.5)으로 수렴 (오차 7.0e-06)
4. τ가 4 → 0.25로 줄수록 arg-max까지의 평균 거리가 단조 감소
   (0.374 → 0.361 → 0.298 → 0.185 → 0.099)
5. τ ∈ [0.25, 4] 전 구간에서 그래디언트가 유한

TF 2.21(Keras 3) 환경에서 순수 TF 연산으로 검증했습니다. 이 저장소가 고정한
TF 2.15(Keras 2)와 무관한 연산들이라 버전에 영향받지 않습니다. 다만 Keras 레이어
자체의 조립은 검증하지 못했으니, 2단계를 밤새 돌리기 전에 한 번만 짧게 확인하세요:

```bash
python3 de_train.py --splits splits.json --head softargmax --fraction 10 \
        --steps 40 --eval-every 20 --tau 0.5 --coord-weight 20 --outdir /tmp/smoke
```

2~3분 안에 끝나고 `/tmp/smoke/softargmax_f10_s0_t0.5_w20.json`이 생기면 통과입니다.

## 보내주실 것

- 1단계: `lr_probe_softargmax.json`
- 2단계: `runs/softargmax_*_t*.json`, `runs/softargmax_*_w*.json` (수십 KB)

받으면 집계표, figure, §5.1 대체 문단까지 작성하겠습니다.
