from __future__ import annotations

from pathlib import Path
import struct
import zlib


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fig1.png"

WIDTH = 1800
HEIGHT = 1050

WHITE = (255, 255, 255, 255)
INK = (19, 34, 56, 255)
TEXT = (37, 50, 68, 255)
MUTED = (79, 93, 115, 255)
BORDER = (47, 59, 82, 255)
ARROW = (74, 88, 114, 255)
BLUE = (238, 246, 255, 255)
GOLD = (255, 247, 232, 255)
GREEN = (237, 249, 241, 255)
GRAY = (245, 247, 250, 255)

FONT: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10011", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00001", "00001", "00001", "00001", "10001", "10001", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


class Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(bg * width * height)

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 4
            self.pixels[idx : idx + 4] = bytes(color)

    def fill_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int, int]) -> None:
        for yy in range(max(0, y), min(self.height, y + h)):
            row_start = (yy * self.width + max(0, x)) * 4
            row_end = (yy * self.width + min(self.width, x + w)) * 4
            count = (row_end - row_start) // 4
            self.pixels[row_start:row_end] = bytes(color) * count

    def draw_rect(self, x: int, y: int, w: int, h: int, fill: tuple[int, int, int, int], border: tuple[int, int, int, int], border_w: int = 3) -> None:
        self.fill_rect(x, y, w, h, fill)
        self.fill_rect(x, y, w, border_w, border)
        self.fill_rect(x, y + h - border_w, w, border_w, border)
        self.fill_rect(x, y, border_w, h, border)
        self.fill_rect(x + w - border_w, y, border_w, h, border)

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int, int], width: int = 3) -> None:
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            self.fill_rect(x1 - width // 2, y1 - width // 2, width, width, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def draw_arrow(self, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int, int]) -> None:
        self.draw_line(x1, y1, x2, y2, color, width=4)
        if abs(x2 - x1) >= abs(y2 - y1):
            direction = 1 if x2 > x1 else -1
            self.draw_line(x2, y2, x2 - 18 * direction, y2 - 10, color, width=4)
            self.draw_line(x2, y2, x2 - 18 * direction, y2 + 10, color, width=4)
        else:
            direction = 1 if y2 > y1 else -1
            self.draw_line(x2, y2, x2 - 10, y2 - 18 * direction, color, width=4)
            self.draw_line(x2, y2, x2 + 10, y2 - 18 * direction, color, width=4)

    def draw_char(self, x: int, y: int, ch: str, color: tuple[int, int, int, int], scale: int) -> int:
        pattern = FONT.get(ch.upper(), FONT[" "])
        for row, bits in enumerate(pattern):
            for col, bit in enumerate(bits):
                if bit == "1":
                    self.fill_rect(x + col * scale, y + row * scale, scale, scale, color)
        return 6 * scale

    def draw_text(self, x: int, y: int, text: str, color: tuple[int, int, int, int], scale: int) -> None:
        cursor = x
        for ch in text:
            cursor += self.draw_char(cursor, y, ch, color, scale)

    def text_width(self, text: str, scale: int) -> int:
        return len(text) * 6 * scale

    def draw_centered_text(self, cx: int, y: int, text: str, color: tuple[int, int, int, int], scale: int) -> None:
        self.draw_text(cx - self.text_width(text, scale) // 2, y, text, color, scale)

    def save_png(self, path: Path) -> None:
        raw = bytearray()
        stride = self.width * 4
        for y in range(self.height):
            raw.append(0)
            start = y * stride
            raw.extend(self.pixels[start : start + stride])

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(chunk(b"IHDR", struct.pack("!IIBBBBB", self.width, self.height, 8, 6, 0, 0, 0)))
        png.extend(chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
        png.extend(chunk(b"IEND", b""))
        path.write_bytes(png)


def draw_box(
    canvas: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body: list[str],
    fill: tuple[int, int, int, int],
    *,
    title_scale: int = 3,
    body_scale: int = 2,
) -> None:
    canvas.draw_rect(x, y, w, h, fill, BORDER, border_w=4)
    canvas.draw_centered_text(x + w // 2, y + 18, title, INK, title_scale)
    body_y = y + 74
    for line in body:
        canvas.draw_centered_text(x + w // 2, body_y, line, TEXT, body_scale)
        body_y += 28


def main() -> None:
    c = Canvas(WIDTH, HEIGHT, WHITE)

    c.draw_centered_text(WIDTH // 2, 35, "ALTERON ARCHITECTURE", INK, 4)
    c.draw_centered_text(WIDTH // 2, 85, "BEHAVIORAL REGRESSION TESTING ACROSS VERSIONED NLP MODEL UPDATES", MUTED, 2)

    c.draw_text(70, 150, "INPUTS", MUTED, 3)
    c.draw_text(560, 150, "PIPELINE STAGES", MUTED, 3)
    c.draw_text(1140, 150, "ARTIFACTS", MUTED, 3)

    draw_box(c, 70, 180, 340, 120, "LABELED SOURCE DATA", ["SA / NLI / TOPIC EXAMPLES", "WITH SOURCE LABELS"], BLUE)
    draw_box(c, 70, 360, 340, 120, "MR REGISTRY", ["VERSIONED METAMORPHIC RELATIONS", "AND SEVERITY METADATA"], BLUE)
    draw_box(c, 70, 540, 340, 120, "MODEL INTEGRATION", ["MODEL LOADER", "OLD VERSION + NEW VERSION"], BLUE)

    draw_box(c, 560, 180, 420, 160, "1. CORPUS GENERATION", ["APPLY SELECTED MRS", "RUN AUTOMATED CHECKS", "FREEZE VALIDATED PAIRS"], GOLD)
    draw_box(c, 560, 410, 420, 160, "2. SNAPSHOT CREATION", ["VERIFY CORPUS HASHES", "RUN OLD AND NEW VERSION INFERENCE", "RECORD PREDICTIONS AND MR OUTCOMES"], GOLD)
    draw_box(c, 560, 640, 420, 160, "3. REGRESSION DIFFERENCING", ["MATCHED-SUBSET COMPARISON", "VERSION-TO-VERSION PASS-RATE DELTAS", "BLOCKING VS NON-BLOCKING FLAGS"], GOLD)

    draw_box(c, 1080, 180, 620, 105, "CONTINUOUS INTEGRATION", ["RUNS THE VERSION-TO-VERSION CHECK BEFORE RELEASE"], GRAY)
    draw_box(c, 1080, 315, 620, 100, "FROZEN CORPUS", ["PER-MR CSVS + SHA-256 MANIFEST", "MANUAL-VALIDATION SAMPLES"], GREEN)
    draw_box(c, 1080, 470, 280, 105, "OLD VERSION SNAPSHOT", ["SOURCE + FOLLOW-UP", "PREDICTIONS"], GREEN, title_scale=2)
    draw_box(c, 1420, 470, 280, 105, "NEW VERSION SNAPSHOT", ["SOURCE + FOLLOW-UP", "PREDICTIONS"], GREEN, title_scale=2)
    draw_box(c, 1080, 700, 620, 100, "REGRESSION REPORT AND CI SUMMARY", ["RELEASE-BLOCKING DECISION", "MACHINE-READABLE CI OUTCOME"], GREEN, title_scale=2)

    c.draw_arrow(410, 240, 560, 240, ARROW)
    c.draw_arrow(410, 420, 560, 260, ARROW)
    c.draw_arrow(410, 600, 560, 490, ARROW)
    c.draw_arrow(980, 260, 1080, 365, ARROW)
    c.draw_arrow(980, 490, 1080, 520, ARROW)
    c.draw_arrow(980, 490, 1420, 520, ARROW)
    c.draw_arrow(1220, 575, 700, 700, ARROW)
    c.draw_arrow(1560, 575, 840, 700, ARROW)
    c.draw_arrow(980, 720, 1080, 750, ARROW)

    c.save_png(OUTPUT)


if __name__ == "__main__":
    main()
