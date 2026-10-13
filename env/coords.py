"""월드 좌표계 — 사용자 설정 / GUI 표시 / RL 관측의 위치값이 모두 이 좌표계를 쓴다.

    일반 직교좌표:  x = 가로축 (오른쪽 +),  y = 세로축 (위쪽 +)
    원점 (0, 0)   = 맵 중앙 셀. 중앙이 셀 경계에 걸리는 짝수 크기는 반올림
                    -> 원점 셀 인덱스 row0 = rows // 2, col0 = cols // 2
                    (예: 40 x 60 맵 -> 중앙 (19.5, 29.5) -> 반올림 (20, 30))

내부 그리드 배열은 numpy 인덱싱을 위해 (row, col) 그대로 쓴다 (row 는 아래로 증가).
변환은 반드시 이 모듈의 WorldFrame 으로만 한다:
    cell (row, col)  <->  xy (x, y)
    x = col - col0,   y = row0 - row
"""


class WorldFrame:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.row0 = rows // 2
        self.col0 = cols // 2

    def to_xy(self, cell):
        r, c = cell
        return (c - self.col0, self.row0 - r)

    def to_cell(self, xy):
        x, y = xy
        return (self.row0 - int(y), self.col0 + int(x))

    def normalized(self, cell):
        """RL 관측용 정규화 위치 (x / cols, y / rows) — 대략 [-0.5, 0.5], 맵 크기와 무관."""
        x, y = self.to_xy(cell)
        return (x / self.cols, y / self.rows)

    def extent(self):
        """맵이 덮는 x, y 범위 (x_min, x_max, y_min, y_max)."""
        return (-self.col0, self.cols - 1 - self.col0, self.row0 - (self.rows - 1), self.row0)
