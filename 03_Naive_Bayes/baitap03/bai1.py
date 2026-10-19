# -*- coding: utf-8 -*-
"""Bài 1: định lý Bayes với ba nhóm sinh viên, mở rộng từ d1."""


def bayes_nhieu_nhanh(tien_nghiem, kha_nang):
    """Tính P(nhóm | qua môn) bằng công thức xác suất toàn phần."""
    trong_so = [p * q for p, q in zip(tien_nghiem, kha_nang)]
    bang_chung = sum(trong_so)
    return bang_chung, [w / bang_chung for w in trong_so]


def main():
    cac_nhom = ["Hoc deu", "Hoc don", "It hoc"]
    tien_nghiem = [0.50, 0.30, 0.20]
    ty_le_qua = [0.95, 0.70, 0.30]
    p_qua, hau_nghiem = bayes_nhieu_nhanh(tien_nghiem, ty_le_qua)

    print(f"P(qua mon) = 0.50*0.95 + 0.30*0.70 + 0.20*0.30 = {p_qua:.4f}")
    print(f"{'Nhom':12s} {'P(nhom)':>10s} {'P(qua|nhom)':>14s} {'P(nhom|qua)':>14s}")
    for ten, p, q, h in zip(cac_nhom, tien_nghiem, ty_le_qua, hau_nghiem):
        print(f"{ten:12s} {p:10.4f} {q:14.4f} {h:14.6f}")
    print(f"Tong ba xac suat hau nghiem = {sum(hau_nghiem):.6f}")
    # Kiểm tra kết quả là một phân phối xác suất hợp lệ.
    assert abs(sum(hau_nghiem) - 1.0) < 1e-12


if __name__ == "__main__":
    main()
