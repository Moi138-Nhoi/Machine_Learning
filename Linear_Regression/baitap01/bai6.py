
w = 0.078367
b = 0.401752


def du_doan_gia(dien_tich):

    if dien_tich < 35.5 or dien_tich > 117.5:
        print("CANH BAO: Dien tich nam ngoai khoang du lieu da hoc.")

    gia_du_doan = w * dien_tich + b

    return gia_du_doan


gia_60 = du_doan_gia(60)
print(f"60 m2 -> {gia_60:.3f} ty dong")

print()

gia_80 = du_doan_gia(80)
print(f"80 m2 -> {gia_80:.3f} ty dong")

print()

gia_200 = du_doan_gia(200)
print(f"200 m2 -> {gia_200:.3f} ty dong")