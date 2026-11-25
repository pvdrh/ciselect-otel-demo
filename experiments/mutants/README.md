# Bộ sinh dữ liệu mutation cho `opentelemetry-demo` (ts02)

Pipeline sinh bộ dữ liệu D1: đột biến code của từng service, dựng lại demo, chạy bộ test telemetry, ghi kết quả từng test. Thiết kế xem `docs/preregistration.md` (C1, C2) và `plans/2026-09-30-d1-sinh-bo-du-lieu.md`. Mô tả dữ liệu đầu ra xem `docs/2026-10-02-data-card/data-card.md`.

Các bước dưới đây ghi đúng cách đã chạy run `results/ts02_dataset/20261001-213236_seed0` (D1-7).

## 1. Yêu cầu máy

| Mục | Đã dùng | Ghi chú |
|---|---|---|
| OS | Windows 10 Home + Git Bash | Lệnh viết cho Git Bash. `make_patches.py` gọi `sh ./docker-gen-proto.sh` và đặt `MSYS_NO_PATHCONV=1`. Chỉ chạy trên Windows: `batch.py:73` kiểm dung lượng trống của ổ C bằng `shutil.disk_usage("C:/")`; chạy trên Linux phải sửa dòng này, và chưa thử |
| RAM | 16 GB; Docker Desktop cấp 8,2 GB, 8 CPU | Đỉnh RAM cả demo khoảng 3,7 GB. Chạy **một** demo mỗi lúc |
| Đĩa | Ổ C còn ≥ 8 GB trong suốt quá trình chạy | `batch.py` dừng (`LowDisk`) khi ổ C còn dưới 8 GB (`MIN_FREE_GB`). Ổ C được kiểm cứng trong code (`batch.py:73`). Image demo chiếm khoảng chục GB |
| Docker | 29.4.2, có `docker compose` | |
| Python | 3.10.0 + `pyyaml` (đã dùng 6.0.3) | Các module khác đều thuộc stdlib. Test cần `pytest` |
| Mạng | Kéo image từ `ghcr.io` | Sau mỗi mutant có build, runner kéo lại image phát hành của service đó |

## 2. Chuẩn bị (làm một lần)

```bash
W=/e/luan_van-02-ci-test-selection          # gốc worktree; đổi theo máy
DEMO=$W/data/raw/opentelemetry-demo

# 2.1 Clone demo đúng tag 3.1.0 (commit dedc0178918e260823323b8d95005a8cb924b007)
#     Clone nông, như khi sinh D1: git viết tắt hash ở dòng `index` của patch theo số object của repo, nên clone có
#     lịch sử cho patch_sha256 khác dù nội dung patch trùng
git clone --depth 1 --branch 3.1.0 https://github.com/open-telemetry/opentelemetry-demo $DEMO
git -C $DEMO checkout 3.1.0
git -C $DEMO status --short                  # phải rỗng: runner áp patch rồi `git checkout -- .` trên clone này

# 2.2 Kéo image phát hành 3.1.0 của demo
cd $DEMO
DEMO_VERSION=3.1.0 docker compose --env-file .env \
  -f compose.yaml -f compose.observability.yaml -f compose.extras.yaml pull

# 2.3 Build image chạy test telemetry (cách của target run-telemetry-tests-minimal trong Makefile của demo)
docker build -t opentelemetry-demo-telemetry-tests ./test/telemetry

# 2.4 Kiểm tra pipeline
cd $W/experiments && python -m pytest mutants ts02 graph spike -q
```

**Cấu hình cố định do runner tự truyền, không cần sửa tay** (`experiments/spike/run_spike.py`, `experiments/ts02/batch.py`):
- Compose: `compose.yaml` + `compose.observability.yaml` + `compose.extras.yaml` + `experiments/spike/compose.memfix.yaml`. File cuối nâng giới hạn RAM của checkout và product-catalog lên 64M, `GOMEMLIMIT=48MiB` (#31).
- Container test: `TEST_SCOPE=minimal`, `WARMUP_SECONDS=240` (tối đa, dừng sớm khi backend có dữ liệu), `POLL_TIMEOUT=60`, `WARMUP_PROBE_ENABLED=false`, không chạy Cypress.
- Mỗi lần chạy: `up --wait` (timeout 600 s) → chạy test → `down -v --remove-orphans` → `git checkout -- .` → kéo lại image của service vừa build.

## 3. Đo thời gian build (D1-0, làm một lần)

Đầu vào cho chi phí `build_nocache(s)` của tiền đăng ký C0.

```bash
cd $W/experiments/ts02
python build_times.py $W/results/ts02_build_times/<YYYYMMDD-HHMMSS>_seed0 3
```

## 4. Sinh dữ liệu

```bash
OUT=$W/results/ts02_dataset/<YYYYMMDD-HHMMSS>_seed0
mkdir -p $OUT/mutants
cd $W/experiments/mutants
python sites.py $DEMO $OUT/mutants/sites.csv > $OUT/mutants/sites_summary.txt   # liệt kê vị trí có thể đột biến
python make_patches.py $DEMO $OUT/mutants/sites.csv $OUT/mutants 380 0 > $OUT/make_patches.log 2>&1
ls $OUT/mutants/patches | wc -l                                    # phải ra 380

cd $W/experiments/ts02
python batch.py $OUT/mutants $OUT 20 > $OUT/batch.log 2>&1         # 20 = chèn 1 lần chạy sạch sau mỗi 20 mutant
python build_labels.py $OUT                                         # → labels.csv, mutant_status.csv
```

- **Tham số:** `380 0` là N và seed (#36). Tỉ lệ loại mutant M0/M1/M2/M3 = 0,2/0,3/0,3/0,2 nằm cứng trong `make_patches.py` (#29).
- **Chạy tiếp được khi bị ngắt:** chạy lại đúng lệnh `batch.py`; mutant đã có trong `done.csv` sẽ được bỏ qua. Trước khi chạy lại, kiểm tra `git -C $DEMO status --short` rỗng. Nếu không rỗng thì một patch còn sót do bị ngắt giữa chừng, `git -C $DEMO checkout -- .` rồi mới chạy.
- **Theo dõi tiến độ:** `wc -l < $OUT/done.csv`, `tail $OUT/batch.log`. Mỗi dòng log có dạng `<run_id> <status> <số test fail>`.
- **Thời gian đo được trên máy ở §1** (tổng `t_total` trong `table.csv` của run): 380 mutant tốn 70,3 giờ máy, khoảng 11,1 phút mỗi mutant, tính cả lần chạy lại và lần chạy sạch. Trung vị một lần chạy: 9,2 phút với mutant làm test fail (chạy 2 lần), 3,1 phút với mutant không làm test fail, 2,1 phút với lần chạy sạch.
- `params.json` và `env.txt` của run ghi lại tham số và môi trường. Không sửa tay; sai thì chạy run mới.

## 5. Kiểm tra sau khi chạy

- Patch sinh lại khớp bản đã lưu: so cột `patch_sha256` trong `mutants.csv`.
- Các lần chạy sạch (`clean-*`) phải có 0 test fail.
- M0 (chỉ đổi comment) phải không có test fail.
- Lần chạy a1 và a2 của cùng một mutant phải cho cùng kết luận. Lệch nhau thì đó là test flaky; quy tắc nhãn xử lý theo C2.

## 6. Chỗ gắn cứng vào `opentelemetry-demo` (giới hạn "chưa tổng quát")

Muốn dùng cho hệ khác phải sửa các chỗ sau:

| Chỗ | File |
|---|---|
| Danh sách service, file handler, tên service trong proto, cách sinh lại stub | `sites.py` (`SERVICES`) |
| Toán tử theo ngôn ngữ (đảo điều kiện, comment, kiểu proto, khoá JSON của quote/email/shipping) | `sites.py` |
| Đột biến biến môi trường (M3) đọc từ `compose.yaml` + `.env` | `sites.py` (`env_sites`), `experiments/graph/extract.py` |
| Sinh lại stub proto bằng `docker-gen-proto.sh` của demo | `make_patches.py` |
| Đường dẫn demo, tag, các file compose, image test, biến môi trường của oracle | `experiments/spike/run_spike.py` |
| Ánh xạ test sang service đọc từ ID test (`test_x::…[svc]`, `[P->C]`) | `build_labels.py`, tiền đăng ký C1 |
| Ổ đĩa kiểm tra dung lượng (`C:/`) | `batch.py` |

## 7. Phạm vi service

Đột biến 14 service ứng dụng có test telemetry. Hai service có `build:` trong `compose.yaml` bị loại vì không test nào chạm tới (D1 ruling 2; đã kiểm ngày 2026-10-02):
- `flagd-ui`
- `telemetry-docs`

Cách kiểm: trong 56 test của scope minimal, các service xuất hiện trong ID test đúng là 14 service này. `SIGNAL_MATRIX` trong `test/telemetry/services.py` của demo cũng không có hai service trên.
