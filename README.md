# ODRA Strategy: Optimal Dynamic Reset Allocation for Uniswap v3

Dự án này triển khai chiến lược ODRA (Optimal Dynamic Reset Allocation) cho Liquidity Provider trên Uniswap v3, sử dụng dữ liệu tick-level thực và deep reinforcement learning.

## Tổng quan

ODRA là một chiến lược cung cấp thanh khoản động:
- Tối ưu hóa phạm vi vị thế dựa trên biến động giá và mẫu khối lượng giao dịch
- Sử dụng deep reinforcement learning để học chính sách tái cân bằng tối ưu
- Tích hợp dữ liệu tick-level thực từ các pool Uniswap v3
- Xem xét chi phí giao dịch và hành vi cạnh tranh của LP

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- pip
- virtualenv hoặc conda
- Git

### Các bước cài đặt

1. Clone repository:
```bash
git clone https://github.com/BinhOiDungNghien/uniswap-lp.git
cd uniswap-lp
```

2. Tạo và kích hoạt môi trường ảo:
```bash
# Sử dụng virtualenv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# HOẶC
venv\Scripts\activate     # Windows

# HOẶC sử dụng conda
conda create -n odra python=3.8
conda activate odra
```

3. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
odra_strategy/
├── config/
│   └── odra_config.yml     # Cấu hình tham số
├── data/
│   ├── raw/                # Dữ liệu tick-level
│   ├── processed/          # Dữ liệu đã xử lý
│   └── utils/             # Tiện ích xử lý dữ liệu
├── features/
│   └── feature_engine.py   # Tính toán đặc trưng
├── model/
│   ├── network.py         # Kiến trúc neural network
│   ├── trainer.py         # Logic huấn luyện
│   └── loss.py           # Hàm loss (CARA utility)
├── strategy/
│   ├── strategy_simulator.py # Mô phỏng chiến lược
│   └── tick_math.py       # Tính toán tick Uniswap v3
└── notebooks/             # Jupyter notebooks phân tích
```

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

Đặt các file dữ liệu tick-level vào thư mục `odra_strategy/data/raw/`. Dữ liệu phải có định dạng:
- Tên file: `arbitrum-[pool_address]-[date].tick.csv`
- Cấu trúc: `timestamp,type,sqrtPriceX96,tick,amount0,amount1,liquidity,position_id`

### 2. Cấu hình tham số

Chỉnh sửa file `odra_strategy/config/odra_config.yml`:
```yaml
# Tham số chiến lược
tau: 3                  # Ngưỡng reset (ticks)
alpha_ewma: 0.05        # Hệ số suy giảm EWMA
tick_spacing: 60        # Khoảng cách tick của pool

# Tham số mô hình
model:
  hidden_layers: 5      # Số lớp ẩn
  hidden_units: 16      # Số unit mỗi lớp
  learning_rate: 0.001  # Tốc độ học
```

### 3. Huấn luyện mô hình

```bash
# Xử lý dữ liệu thô
python -m odra_strategy.main --config odra_strategy/config/odra_config.yml --mode process

# Huấn luyện mô hình
python -m odra_strategy.main --config odra_strategy/config/odra_config.yml --mode train

# Đánh giá chiến lược
python -m odra_strategy.main --config odra_strategy/config/odra_config.yml --mode evaluate
```

Các tham số:
- `--config`: Đường dẫn đến file cấu hình
- `--mode`: Chế độ chạy (process/train/evaluate)

### 4. Phân tích kết quả

Các notebook phân tích được cung cấp trong thư mục `notebooks/`:
- `train_odra_colab.ipynb`: Huấn luyện trên Google Colab
- `kaggle_workflow.ipynb`: Pipeline huấn luyện trên Kaggle

## Theo dõi và gỡ lỗi

1. Log files được lưu trong `odra_strategy/outputs/logs/`
2. Model checkpoints trong `odra_strategy/outputs/models/`
3. Biểu đồ phân tích trong `odra_strategy/outputs/plots/`

## Tài liệu tham khảo

- [Uniswap v3 Whitepaper](https://uniswap.org/whitepaper-v3.pdf)
- [Uniswap v3 Core Contracts](https://github.com/Uniswap/v3-core)

## License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## Citation

```bibtex
@misc{odra2024,
  author = {Le Ngoc Binh},
  title = {ODRA: Optimal Dynamic Reset Allocation for Uniswap v3},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/BinhOiDungNghien/uniswap-lp}
}
``` 