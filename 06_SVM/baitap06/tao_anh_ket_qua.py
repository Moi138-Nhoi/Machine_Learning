"""Tạo ảnh kết quả từ log stdout đã chạy của sáu bài."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


thu_muc = Path(__file__).resolve().parent
font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 18)
le = 24
cao_dong = 28

for so_bai in range(1, 7):
    log = thu_muc / f"ket_qua_bai{so_bai}.txt"
    cac_dong = log.read_text(encoding="utf-8-sig").splitlines()
    tieu_de = f"> python baitap06/bai{so_bai}.py"
    rong = int(max(font.getlength(dong) for dong in [tieu_de, *cac_dong]) + 2 * le)
    cao = (len(cac_dong) + 1) * cao_dong + 2 * le
    anh = Image.new("RGB", (rong, cao), "#15191e")
    ve = ImageDraw.Draw(anh)
    ve.text((le, le), tieu_de, font=font, fill="#8ed1a5")
    for chi_so, dong in enumerate(cac_dong, start=1):
        ve.text((le, le + chi_so * cao_dong), dong, font=font, fill="#e6edf3")
    dau_ra = thu_muc / f"ket_qua_bai{so_bai}.png"
    anh.save(dau_ra)
    print(dau_ra)
