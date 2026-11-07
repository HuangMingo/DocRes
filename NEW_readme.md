# DocRes — phục hồi ảnh tài liệu đa tác vụ

Repo này là mã nguồn chính thức của bài báo **“DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks”** (CVPR 2024). DocRes dùng một mô hình chung để xử lý nhiều dạng suy giảm của ảnh tài liệu thay vì cần một mô hình riêng cho từng tác vụ.

Các tác vụ được hỗ trợ:

- `dewarping`: làm phẳng ảnh tài liệu bị cong/méo phối cảnh;
- `deshadowing`: loại bỏ bóng đổ;
- `appearance`: cải thiện hình thức, ánh sáng và màu sắc;
- `deblurring`: khử nhòe;
- `binarization`: nhị phân hóa ảnh tài liệu;
- `end2end`: chạy tuần tự `dewarping` → `deshadowing` → `appearance`.

Mã nguồn bài báo: <https://github.com/ZZZHANG-jx/DocRes>  
Bài báo: <https://arxiv.org/abs/2405.04408>

## Cấu trúc chính

```text
DocRes/
├── inference.py          # Chạy phục hồi một ảnh hoặc cả thư mục ảnh
├── eval.py               # Chạy dự đoán và tính metric trên tập đánh giá
├── train.py              # Huấn luyện phân tán bằng PyTorch DDP
├── start_train.sh        # Lệnh mẫu để huấn luyện trên 8 GPU
├── requirements.txt      # Các thư viện Python chính
├── utils.py              # Hàm tiện ích, metric và xử lý ảnh
├── models/               # Kiến trúc Restormer/DocRes
├── loaders/              # Dataset loader dùng khi huấn luyện
├── data/
│   ├── MBD/              # Mô hình tạo mask cho tác vụ dewarping
│   ├── preprocess/       # Công cụ chuẩn bị dữ liệu
│   └── README.md         # Hướng dẫn cấu trúc dữ liệu chi tiết
├── input/                # Ảnh mẫu đầu vào
└── restorted/            # Kết quả suy luận (tên thư mục giữ nguyên theo repo)
```

## Yêu cầu môi trường

Cấu hình gốc của repo:

- Python 3.7 (khuyến nghị để tương thích tốt với bộ thư viện cũ);
- PyTorch 1.11.0 và torchvision 0.12.0;
- CUDA 11.3;
- GPU NVIDIA được khuyến nghị. Phần lớn luồng suy luận dùng FP16 và huấn luyện dùng NCCL/DDP, nên CPU không phải cấu hình chạy thực tế phù hợp.

> `requirements.txt` dùng wheel PyTorch CUDA 11.3. Nếu máy dùng CUDA/PyTorch khác, hãy cài cặp `torch`/`torchvision` phù hợp trước, sau đó cài các thư viện còn lại.

## Cài đặt

Chạy các lệnh sau từ thư mục gốc của repo.

### Linux hoặc WSL

```bash
python3.7 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.7 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra PyTorch có nhận GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Tải trọng số mô hình

Cần hai checkpoint:

1. Tải `mbd.pkl` từ [liên kết checkpoint của tác giả](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK), đặt tại:

   ```text
   data/MBD/checkpoint/mbd.pkl
   ```

2. Tải `docres.pkl` từ cùng liên kết, đặt tại:

   ```text
   checkpoints/docres.pkl
   ```
Tạo thư mục nếu chưa có:

```bash
mkdir -p data/MBD/checkpoint checkpoints restorted
```

PowerShell tương đương:

```powershell
New-Item -ItemType Directory -Force data/MBD/checkpoint, checkpoints, restorted
```

Sau bước này, cấu trúc tối thiểu cần có là:

```text
data/MBD/checkpoint/mbd.pkl
checkpoints/docres.pkl
```

## Chạy suy luận

Ví dụ làm phẳng ảnh mẫu và lưu cả ba DTSPrompt:

```bash
python inference.py \
  --im_path ./input/for_dewarping.png \
  --task dewarping \
  --save_dtsprompt 1
```

Trên PowerShell có thể viết một dòng:

```powershell
python inference.py --im_path ./input/for_dewarping.png --task dewarping --save_dtsprompt 1
```

Kết quả mặc định được ghi vào `./restorted/`. Ví dụ trên tạo file `restorted/for_dewarping_dewarping.png` và, khi `--save_dtsprompt 1`, thêm ba ảnh prompt.

Các tham số chính:

| Tham số | Ý nghĩa | Giá trị mặc định |
|---|---|---|
| `--im_path` | Đường dẫn một ảnh hoặc một thư mục ảnh | `./distorted/` |
| `--task` | Một trong 6 tác vụ nêu trên | `dewarping` |
| `--model_path` | Checkpoint DocRes | `./checkpoints/docres.pkl` |
| `--out_folder` | Thư mục kết quả; phải tồn tại trước khi chạy | `./restorted/` |
| `--save_dtsprompt` | `1` để lưu các prompt trung gian, `0` để không lưu | `0` |

Một số ví dụ khác:

```bash
# Xóa bóng
python inference.py --im_path ./input/for_deshadowing.jpg --task deshadowing

# Khử nhòe
python inference.py --im_path ./input/for_debluring.png --task deblurring

# Nhị phân hóa
python inference.py --im_path ./input/for_binarization.png --task binarization

# Chạy chuỗi phục hồi end-to-end
python inference.py --im_path ./input/for_end2end.png --task end2end

# Xử lý tất cả file trực tiếp trong một thư mục
python inference.py --im_path ./input --task appearance --out_folder ./restorted
```

Lưu ý: chế độ thư mục không lọc phần mở rộng; nên chỉ để các file ảnh hợp lệ trong thư mục đầu vào.

## Đánh giá

Chuẩn bị dữ liệu theo `data/README.md`. Mỗi tập nằm tại `data/eval/<ten_tap>/`, ảnh đầu vào có hậu tố `_in` và ground truth có hậu tố `_gt`, ví dụ:

```text
data/eval/realdae/001_in.png
data/eval/realdae/001_gt.png
```

Chạy đánh giá:

```bash
python eval.py --dataset realdae
```

Các tên tập hợp lệ và tác vụ tương ứng:

| Dataset | Tác vụ |
|---|---|
| `dir300` | dewarping |
| `kligler`, `jung`, `osr` | deshadowing |
| `docunet_docaligner`, `realdae` | appearance |
| `tdd` | deblurring |
| `dibco18` | binarization |

Script ghi dự đoán cạnh dữ liệu đầu vào với hậu tố `_docres`, sau đó tính PSNR/SSIM hoặc metric nhị phân hóa. Riêng `dir300`, script chỉ sinh kết quả dewarping rồi thoát, không tính metric.

## Huấn luyện

Huấn luyện cần Linux/WSL, nhiều GPU NVIDIA và backend NCCL. Trước khi chạy:

1. Chuẩn bị các tập dữ liệu theo `data/README.md`.
2. Sửa `datasets_setting` trong `train.py`; các đường dẫn hiện tại là đường dẫn tuyệt đối trên máy của tác giả.
3. Sửa `self.background_paths` và `self.shadow_paths` trong `loaders/docres_loader.py` cho dữ liệu của bạn.
4. Cài thêm PIQ vì `train.py` có import thư viện này nhưng nó chưa được liệt kê trong `requirements.txt`:

   ```bash
   pip install piq
   ```

5. Điều chỉnh số GPU trong `start_train.sh`. File gốc cấu hình 8 GPU (`CUDA_VISIBLE_DEVICES=0,...,7` và `--nproc_per_node 8`).

Chạy cấu hình mẫu:

```bash
bash start_train.sh
```

Hoặc chạy trực tiếp, ví dụ với 2 GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --master_port 26413 \
  train.py \
  --experiment_name docres_experiment \
  --batch_size 10 \
  --total_iter 100000
```

Checkpoint và log được lưu trong `checkpoints/<experiment_name>/` theo mặc định.

## Lưu ý về mã nguồn hiện tại

- Hãy chạy lệnh tại thư mục gốc của repo vì nhiều đường dẫn trong code là đường dẫn tương đối.
- `train.py --resume` hiện tham chiếu biến `checkpoint` trước khi nạp file; cần sửa logic nạp checkpoint nếu muốn tiếp tục một phiên huấn luyện.
- Tùy chọn `--tboard` dùng `SummaryWriter` nhưng file hiện chưa import lớp này. Nếu cần TensorBoard, cài `tensorboard` và thêm `from torch.utils.tensorboard import SummaryWriter` vào `train.py`.
- `requirements.txt` khóa phiên bản PyTorch/CUDA khá cũ; môi trường mới có thể cần điều chỉnh phiên bản hoặc chạy trong container tương thích.
- Repo không cung cấp checkpoint trong mã nguồn; suy luận sẽ không chạy cho đến khi đặt đúng hai file trọng số nói trên.

## Trích dẫn

```bibtex
@inproceedings{zhangdocres2024,
  author    = {Jiaxin Zhang and Dezhi Peng and Chongyu Liu and Peirong Zhang and Lianwen Jin},
  title     = {DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```

Repo được phát hành theo giấy phép trong file `LICENSE`.
