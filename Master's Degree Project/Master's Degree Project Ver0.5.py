import sys
import os
import json
import traceback
from datetime import datetime
from functools import partial
import cv2
import numpy as np
import pandas as pd
from mss import MSS
import pygetwindow as gw

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel,
                             QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# YOLO 가중치 라이브러리 (오프라인 환경 안전 임포트)
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

CONFIG_FILE = "config.json"


class AutoFocusTestApp(QWidget):
    def __init__(self):
        super().__init__()
        self.sct = MSS()
        self.current_frame = None

        # 안정성: UI 시그널 루프 방지
        self.is_loading_config = False

        # 모델별 데이터 및 글로벌 UI 데이터 관리
        self.models_data = {f"model_{i}": self.get_default_model_config() for i in range(1, 11)}
        self.global_presets = {f"preset_{i}": {} for i in range(1, 4)}

        # 캡처 및 검사 기준 변수 초기화
        self.offset_x = 0
        self.offset_y = 0
        self.crop_width_pct = 100
        self.crop_height_pct = 100
        self.use_center_crop = False
        self.grid_offset_x = 0
        self.grid_offset_y = 0

        self.target_score_r = 1000.0
        self.target_score_c = 1000.0
        self.target_score_l = 1000.0
        self.max_allowed_dev_rl = 20.0
        self.min_brightness_limit = 60.0
        self.current_brightness = 0.0

        # YOLO 모델 초기화 (오직 실시간 시각적 가이드용)
        self.yolo_model = None
        if YOLO_AVAILABLE:
            model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
            if os.path.exists(model_path):
                try:
                    self.yolo_model = YOLO(model_path)
                except Exception as e:
                    print(f"YOLO 로드 실패: {e}")

        self.initUI()
        self.load_all_configs_from_file()
        self.load_selected_model()

        # 50ms 간격 화면 갱신
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

    def get_default_model_config(self):
        """모델별 설정은 오직 임계값만 관리합니다."""
        return {
            "target_score_r": 1000.0,
            "target_score_c": 1000.0,
            "target_score_l": 1000.0,
            "max_allowed_dev_rl": 20.0
        }

    def parse_resolution(self, res_str):
        try:
            w, h = map(int, res_str.split('x'))
            return w, h
        except Exception:
            return 1920, 1080

    def block_ui_signals(self, block: bool):
        self.spin_x.blockSignals(block)
        self.spin_y.blockSignals(block)
        self.spin_w.blockSignals(block)
        self.spin_h.blockSignals(block)
        self.spin_grid_x.blockSignals(block)
        self.spin_grid_y.blockSignals(block)
        self.chk_center_crop.blockSignals(block)
        self.combo_res.blockSignals(block)
        self.spin_min_bright.blockSignals(block)
        self.spin_thresh_r.blockSignals(block)
        self.spin_thresh_c.blockSignals(block)
        self.spin_thresh_l.blockSignals(block)
        self.spin_thresh_dev.blockSignals(block)

    def load_all_configs_from_file(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    file_data = json.load(f)

                    # 1. 모델 임계값 로드
                    for i in range(1, 11):
                        m_key = f"model_{i}"
                        if m_key in file_data:
                            self.models_data[m_key].update(file_data[m_key])

                    # 2. 글로벌 프리셋 로드
                    for i in range(1, 4):
                        p_key = f"preset_{i}"
                        if p_key in file_data:
                            self.global_presets[p_key] = file_data[p_key]
            except Exception as e:
                print(f"설정 파일 읽기 오류: {e}")

    def save_configs_to_file(self):
        try:
            merged_data = {**self.models_data, **self.global_presets}
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, indent=4)
        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")

    # =========================================================================
    # [수정됨] 프리셋 관리: 캡처 위치 및 글로벌 설정 저장/불러오기
    # =========================================================================
    def apply_preset_to_ui(self, slot: int):
        self.is_loading_config = True
        slot_key = f"preset_{slot}"
        cfg = self.global_presets.get(slot_key, {})

        if not cfg:
            self.status_label.setText(f"프리셋 #{slot} 데이터가 없습니다.")
            self.is_loading_config = False
            return

        self.block_ui_signals(True)
        if "offset_x" in cfg: self.spin_x.setValue(cfg["offset_x"])
        if "offset_y" in cfg: self.spin_y.setValue(cfg["offset_y"])
        if "crop_width_pct" in cfg: self.spin_w.setValue(cfg["crop_width_pct"])
        if "crop_height_pct" in cfg: self.spin_h.setValue(cfg["crop_height_pct"])
        if "grid_offset_x" in cfg: self.spin_grid_x.setValue(cfg["grid_offset_x"])
        if "grid_offset_y" in cfg: self.spin_grid_y.setValue(cfg["grid_offset_y"])
        if "use_center_crop" in cfg: self.chk_center_crop.setChecked(cfg["use_center_crop"])
        if "min_brightness_limit" in cfg: self.spin_min_bright.setValue(cfg["min_brightness_limit"])

        if "target_resolution" in cfg:
            idx = self.combo_res.findText(cfg["target_resolution"])
            if idx >= 0: self.combo_res.setCurrentIndex(idx)

        self.block_ui_signals(False)
        self.update_runtime_vars_from_ui()
        self.is_loading_config = False

        self.lbl_active_slot.setText(f"현재 선택된 프리셋: #{slot}")
        self.status_label.setText(f"프리셋 #{slot} 불러오기 성공 (캡처 설정 갱신)")

    def save_ui_to_preset(self, slot: int):
        slot_key = f"preset_{slot}"
        self.global_presets[slot_key] = {
            "offset_x": self.spin_x.value(),
            "offset_y": self.spin_y.value(),
            "crop_width_pct": self.spin_w.value(),
            "crop_height_pct": self.spin_h.value(),
            "use_center_crop": self.chk_center_crop.isChecked(),
            "grid_offset_x": self.spin_grid_x.value(),
            "grid_offset_y": self.spin_grid_y.value(),
            "target_resolution": self.combo_res.currentText(),
            "min_brightness_limit": self.spin_min_bright.value()
        }
        self.save_configs_to_file()
        self.lbl_active_slot.setText(f"현재 선택된 프리셋: #{slot}")
        self.status_label.setText(f"프리셋 #{slot} 저장 완료!")

    def update_runtime_vars_from_ui(self):
        self.offset_x = self.spin_x.value()
        self.offset_y = self.spin_y.value()
        self.crop_width_pct = self.spin_w.value()
        self.crop_height_pct = self.spin_h.value()
        self.grid_offset_x = self.spin_grid_x.value()
        self.grid_offset_y = self.spin_grid_y.value()
        self.use_center_crop = self.chk_center_crop.isChecked()

        self.target_resolution = self.combo_res.currentText()
        self.target_w, self.target_h = self.parse_resolution(self.target_resolution)

        self.min_brightness_limit = self.spin_min_bright.value()

        self.target_score_r = self.spin_thresh_r.value()
        self.target_score_c = self.spin_thresh_c.value()
        self.target_score_l = self.spin_thresh_l.value()
        self.max_allowed_dev_rl = self.spin_thresh_dev.value()

    def on_setting_changed(self):
        if self.is_loading_config:
            return
        self.update_runtime_vars_from_ui()

    def get_save_dir(self):
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        target_dir = os.path.join(desktop_path, "Test 결과")
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def apply_pastel_stylesheet(self):
        self.setStyleSheet("""
            QWidget { background-color: #F8FAFC; color: #334155; font-family: 'Malgun Gothic', 'Segoe UI', sans-serif; }
            QGroupBox { font-weight: bold; font-size: 11px; border: 1px solid #E2E8F0; border-radius: 8px; background-color: #FFFFFF; margin-top: 10px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 5px; background-color: #FFFFFF; color: #475569; }
            QPushButton { background-color: #E2E8F0; border: 1px solid #CBD5E1; border-radius: 6px; padding: 5px 10px; font-weight: bold; color: #334155; }
            QPushButton:hover { background-color: #CBD5E1; }
            QSpinBox, QDoubleSpinBox, QComboBox { border: 1px solid #CBD5E1; border-radius: 5px; padding: 3px 6px; background-color: #F8FAFC; color: #1E293B; }
            QCheckBox { font-weight: bold; color: #475569; }
        """)

    def initUI(self):
        self.setWindowTitle("IDIS Focus Inspector - Auto Test")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setGeometry(900, 150, 740, 660)
        self.setMinimumSize(660, 580)

        self.apply_pastel_stylesheet()

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # [좌측: 영상 프리뷰 영역]
        self.image_label = QLabel("창 찾는 중...", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 2px solid #E2E8F0; border-radius: 8px; background-color: #1E293B; color: #94A3B8;")
        self.image_label.setScaledContents(True)
        self.image_label.setMinimumSize(320, 240)
        main_layout.addWidget(self.image_label, stretch=5)

        # [우측: 제어 패널]
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)

        # 모델 관리 UI
        self.model_box = QGroupBox("검사 모델 선택 (임계값 4종 전용)")
        model_layout = QGridLayout()
        self.lbl_active_model = QLabel("현재 선택된 모델: Model 1", self)
        self.lbl_active_model.setAlignment(Qt.AlignCenter)
        model_layout.addWidget(self.lbl_active_model, 0, 0, 1, 3)

        self.combo_models = QComboBox()
        self.combo_models.addItems([f"Model {i}" for i in range(1, 11)])
        # 모델 콤보박스 변경 시 즉각적으로 불러오기 실행
        self.combo_models.currentIndexChanged.connect(self.load_selected_model)
        model_layout.addWidget(self.combo_models, 1, 0)

        btn_save_model = QPushButton("현재 임계값 모델에 저장")
        btn_save_model.setStyleSheet("background-color: #FDE68A; color: #92400E; border: none;")
        btn_save_model.clicked.connect(self.save_selected_model)
        model_layout.addWidget(btn_save_model, 1, 1, 1, 2)

        self.model_box.setLayout(model_layout)
        right_panel.addWidget(self.model_box)

        # 헤더
        self.result_header_label = QLabel("READY", self)
        self.result_header_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.result_header_label)

        # JIG 알람 바
        self.jig_alarm_label = QLabel("JIG 상태 정상", self)
        self.jig_alarm_label.setAlignment(Qt.AlignCenter)
        self.jig_alarm_label.setStyleSheet(
            "color: #0369A1; background-color: #E0F2FE; font-weight: bold; padding: 4px; border-radius: 6px;")
        right_panel.addWidget(self.jig_alarm_label)

        # 검사시작 버튼 (YOLO 버튼 제거됨)
        self.start_btn = QPushButton("Focus 검사 시작", self)
        self.start_btn.setStyleSheet("""
            QPushButton { font-size: 14px; font-weight: bold; background-color: #4F46E5; color: white; padding: 10px; border-radius: 6px; border: none; }
            QPushButton:hover { background-color: #4338CA; }
        """)
        self.start_btn.clicked.connect(self.run_focus_test)
        right_panel.addWidget(self.start_btn)

        # 구역 점수 및 편차율
        self.total_score_label = QLabel("Right: - | Center: - | Left: -", self)
        self.total_score_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.total_score_label)

        self.dev_label = QLabel(f"R-L 편차율: - (허용: {self.max_allowed_dev_rl:.1f}%)", self)
        self.dev_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.dev_label)

        # 글로벌 캡처 설정 프리셋
        self.preset_box = QGroupBox("화면 캡처 설정 프리셋 (#1, #2, #3)")
        preset_layout = QGridLayout()
        preset_layout.setSpacing(4)
        preset_layout.setContentsMargins(6, 12, 6, 6)

        self.lbl_active_slot = QLabel("현재 선택된 프리셋: #1", self)
        self.lbl_active_slot.setAlignment(Qt.AlignCenter)
        self.lbl_active_slot.setStyleSheet(
            "background-color: #E0E7FF; color: #3730A3; font-weight: bold; border-radius: 4px; padding: 3px;")
        preset_layout.addWidget(self.lbl_active_slot, 0, 0, 1, 3)

        for i in range(1, 4):
            btn_load = QPushButton(f"#{i} 불러오기")
            btn_load.setStyleSheet(
                "QPushButton { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; } QPushButton:hover { background-color: #BAE6FD; }")
            btn_load.clicked.connect(partial(self.apply_preset_to_ui, i))
            preset_layout.addWidget(btn_load, 1, i - 1)

            btn_save = QPushButton(f"#{i} 저장")
            btn_save.clicked.connect(partial(self.save_ui_to_preset, i))
            preset_layout.addWidget(btn_save, 2, i - 1)

        self.preset_box.setLayout(preset_layout)
        right_panel.addWidget(self.preset_box)

        # 캡처 영역 패널
        self.capture_box = QGroupBox("캡처 영역 및 위치/배율 설정")
        cap_layout = QGridLayout()
        cap_layout.setSpacing(4)
        cap_layout.setContentsMargins(6, 12, 6, 6)

        cap_layout.addWidget(QLabel("처리 해상도:"), 0, 0)
        self.combo_res = QComboBox()
        self.combo_res.addItems(["1920x1080", "1280x720", "3840x2160"])
        self.combo_res.currentTextChanged.connect(self.on_setting_changed)
        cap_layout.addWidget(self.combo_res, 0, 1, 1, 3)

        cap_layout.addWidget(QLabel("창 위치 X 오프셋:"), 1, 0)
        self.spin_x = QSpinBox()
        self.spin_x.setRange(-1000, 1000)
        self.spin_x.valueChanged.connect(self.on_setting_changed)
        cap_layout.addWidget(self.spin_x, 1, 1)

        cap_layout.addWidget(QLabel("창 위치 Y 오프셋:"), 1, 2)
        self.spin_y = QSpinBox()
        self.spin_y.setRange(-1000, 1000)
        self.spin_y.valueChanged.connect(self.on_setting_changed)
        cap_layout.addWidget(self.spin_y, 1, 3)

        cap_layout.addWidget(QLabel("캡처 폭 비율(%):"), 2, 0)
        self.spin_w = QSpinBox()
        self.spin_w.setRange(10, 100)
        self.spin_w.valueChanged.connect(self.on_setting_changed)
        cap_layout.addWidget(self.spin_w, 2, 1)

        cap_layout.addWidget(QLabel("캡처 높이 비율(%):"), 2, 2)
        self.spin_h = QSpinBox()
        self.spin_h.setRange(10, 100)
        self.spin_h.valueChanged.connect(self.on_setting_changed)
        cap_layout.addWidget(self.spin_h, 2, 3)

        cap_layout.addWidget(QLabel("선 X 이동(Shift):"), 3, 0)
        self.spin_grid_x = QSpinBox()
        self.spin_grid_x.setRange(-500, 500)
        self.spin_grid_x.valueChanged.connect(self.on_setting_changed)
        cap_layout.addWidget(self.spin_grid_x, 3, 1)

        cap_layout.addWidget(QLabel("선 Y 이동(Shift):"), 3, 2)
        self.spin_grid_y = QSpinBox()
        self.spin_grid_y.setRange(-500, 500)
        self.spin_grid_y.valueChanged.connect(self.on_setting_changed)
        cap_layout.addWidget(self.spin_grid_y, 3, 3)

        self.chk_center_crop = QCheckBox("중앙 90% 영역만 분석")
        self.chk_center_crop.stateChanged.connect(self.on_setting_changed)
        cap_layout.addWidget(self.chk_center_crop, 4, 0, 1, 4)

        self.capture_box.setLayout(cap_layout)
        right_panel.addWidget(self.capture_box)

        # AI 조도 측정 패널
        self.ai_box = QGroupBox("조도 실시간 측정 & 알람 설정 (LUX)")
        ai_layout = QGridLayout()
        ai_layout.setSpacing(4)
        ai_layout.setContentsMargins(6, 12, 6, 6)

        ai_layout.addWidget(QLabel("현재 조도(밝기):"), 0, 0)
        self.lbl_current_brightness = QLabel("0.0 LUX", self)
        self.lbl_current_brightness.setStyleSheet("font-weight: bold; color: #4F46E5;")
        ai_layout.addWidget(self.lbl_current_brightness, 0, 1)

        ai_layout.addWidget(QLabel("최저 허용 조도:"), 0, 2)
        self.spin_min_bright = QDoubleSpinBox()
        self.spin_min_bright.setRange(0, 255)
        self.spin_min_bright.setSingleStep(5)
        self.spin_min_bright.setValue(self.min_brightness_limit)
        self.spin_min_bright.valueChanged.connect(self.on_setting_changed)
        ai_layout.addWidget(self.spin_min_bright, 0, 3)

        self.ai_box.setLayout(ai_layout)
        right_panel.addWidget(self.ai_box)

        # 3x3 그리드 점수 패널
        self.group_box = QGroupBox("3x3 영역 점수")
        grid_layout = QGridLayout()
        grid_layout.setSpacing(4)
        grid_layout.setContentsMargins(6, 14, 6, 6)

        self.region_labels = []
        for row in range(3):
            row_labels = []
            for col in range(3):
                lbl = QLabel(f"R{row + 1}{col + 1}\n-", self)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(
                    "border: 1px solid #E2E8F0; background-color: #F8FAFC; border-radius: 4px; font-weight: bold;")
                grid_layout.addWidget(lbl, row, col)
                row_labels.append(lbl)
            self.region_labels.append(row_labels)

        self.group_box.setLayout(grid_layout)
        right_panel.addWidget(self.group_box)

        # 임계값 패널
        self.thresh_box = QGroupBox("구역별 판정 임계값 (모델 연동)")
        thresh_layout = QGridLayout()
        thresh_layout.setSpacing(4)
        thresh_layout.setContentsMargins(6, 12, 6, 6)

        thresh_layout.addWidget(QLabel("Right 임계값:"), 0, 0)
        self.spin_thresh_r = QDoubleSpinBox()
        self.spin_thresh_r.setRange(0, 50000)
        self.spin_thresh_r.setSingleStep(50)
        self.spin_thresh_r.valueChanged.connect(self.on_setting_changed)
        thresh_layout.addWidget(self.spin_thresh_r, 0, 1)

        thresh_layout.addWidget(QLabel("Center 임계값:"), 0, 2)
        self.spin_thresh_c = QDoubleSpinBox()
        self.spin_thresh_c.setRange(0, 50000)
        self.spin_thresh_c.setSingleStep(50)
        self.spin_thresh_c.valueChanged.connect(self.on_setting_changed)
        thresh_layout.addWidget(self.spin_thresh_c, 0, 3)

        thresh_layout.addWidget(QLabel("Left 임계값:"), 1, 0)
        self.spin_thresh_l = QDoubleSpinBox()
        self.spin_thresh_l.setRange(0, 50000)
        self.spin_thresh_l.setSingleStep(50)
        self.spin_thresh_l.valueChanged.connect(self.on_setting_changed)
        thresh_layout.addWidget(self.spin_thresh_l, 1, 1)

        thresh_layout.addWidget(QLabel("R-L 편차허용(%):"), 1, 2)
        self.spin_thresh_dev = QDoubleSpinBox()
        self.spin_thresh_dev.setRange(0, 100)
        self.spin_thresh_dev.setSingleStep(1.0)
        self.spin_thresh_dev.valueChanged.connect(self.on_setting_changed)
        thresh_layout.addWidget(self.spin_thresh_dev, 1, 3)

        self.thresh_box.setLayout(thresh_layout)
        right_panel.addWidget(self.thresh_box)

        self.status_label = QLabel("대기 중", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(self.status_label)

        right_panel.addStretch(1)
        main_layout.addLayout(right_panel, stretch=4)
        self.setLayout(main_layout)
        self.update_ui_fonts()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_ui_fonts()

    def update_ui_fonts(self):
        scale = max(0.75, min(self.width() / 740.0, self.height() / 660.0))
        header_size = max(14, int(18 * scale))
        label_size = max(9, int(10.5 * scale))
        sub_size = max(8, int(9.0 * scale))

        current_text = self.result_header_label.text()
        if "PASS" in current_text:
            header_style = f"color: #065F46; background-color: #D1FAE5; font-size: {header_size}px; font-weight: bold; border-radius: 6px; padding: 5px;"
        elif "FAIL" in current_text:
            header_style = f"color: #991B1B; background-color: #FEE2E2; font-size: {header_size}px; font-weight: bold; border-radius: 6px; padding: 5px;"
        else:
            header_style = f"color: #475569; background-color: #E2E8F0; font-size: {header_size}px; font-weight: bold; border-radius: 6px; padding: 5px;"
        self.result_header_label.setStyleSheet(header_style)

        self.total_score_label.setStyleSheet(f"font-size: {label_size}px; font-weight: bold; color: #1E293B;")
        self.dev_label.setStyleSheet(f"font-size: {sub_size}px; font-weight: bold; color: #64748B;")
        self.status_label.setStyleSheet(f"font-size: {sub_size}px; color: #64748B;")

    def get_autoattool_rect(self):
        try:
            windows = gw.getWindowsWithTitle("Camera test")
            if not windows:
                windows = gw.getWindowsWithTitle("AutoATTool")

            if windows:
                win = windows[0]
                if not win.isMinimized and win.width > 50 and win.height > 50:
                    left = max(0, win.left + self.offset_x)
                    top = max(0, win.top + self.offset_y)
                    width = int(win.width * (self.crop_width_pct / 100.0))
                    height = int(win.height * (self.crop_height_pct / 100.0))
                    return {"top": top, "left": left, "width": max(10, width), "height": max(10, height)}
        except Exception:
            pass
        return None

    def get_grid_lines_100(self, width, height):
        base_w = width // 3
        base_h = height // 3
        x_lines = [0, max(0, min(width, base_w + self.grid_offset_x)),
                   max(0, min(width, base_w * 2 + self.grid_offset_x)), width]
        y_lines = [0, max(0, min(height, base_h + self.grid_offset_y)),
                   max(0, min(height, base_h * 2 + self.grid_offset_y)), height]
        return x_lines, y_lines

    def calculate_brightness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def calculate_focus_score(self, cell_gray):
        if cell_gray.size == 0: return 0.0
        # 흰색 배경 여백 제외하여 순수 차트 패턴부만 점수화 (안정성 유지)
        _, bg_mask = cv2.threshold(cell_gray, 200, 255, cv2.THRESH_BINARY_INV)
        if cell_gray[bg_mask > 0].size == 0: return 0.0

        blurred = cv2.GaussianBlur(cell_gray, (3, 3), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobel_x, sobel_y)
        sobel_score = np.mean(sobel_mag[bg_mask > 0]) if np.any(bg_mask > 0) else 0.0

        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian_score = np.var(laplacian[bg_mask > 0]) if np.any(bg_mask > 0) else 0.0
        return float((sobel_score * 0.7) + (laplacian_score * 0.3))

    def update_frame(self):
        try:
            capture_area = self.get_autoattool_rect()
            if capture_area is None:
                self.image_label.setText("Camera test 창을 찾을 수 없습니다.")
                return

            full_frame = np.array(self.sct.grab(capture_area))
            base_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGRA2BGR)

            # 1. 조도 실시간 계산 및 UI 갱신
            self.current_brightness = self.calculate_brightness(base_frame)
            self.lbl_current_brightness.setText(f"{self.current_brightness:.1f} LUX")

            cw, ch = self.target_w, self.target_h
            self.current_frame = cv2.resize(base_frame, (cw, ch), interpolation=cv2.INTER_LINEAR)
            preview_img = self.current_frame.copy()

            # 2. YOLO 차트 인식 및 JIG 안착 판정 (conf=0.12로 설정하여 오인식 방지)
            is_yolo_ok = True
            if self.yolo_model is not None:
                results = self.yolo_model(preview_img, conf=0.12, verbose=False)
                boxes = results[0].boxes
                detected_count = len(boxes)

                # 노란색 바운딩 박스 시각화
                if detected_count > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # 📌 안착 판단 기준 개수 (필요시 이 값을 현장 환경에 맞게 조절하세요)
                REQUIRED_CHART_COUNT = 8

                if detected_count < REQUIRED_CHART_COUNT:
                    is_yolo_ok = False
                    yolo_msg = f"차트 정렬/안착 필요 ({detected_count}/24개 감지)"
                else:
                    yolo_msg = f"차트 안착 완료 ({detected_count}/24개 감지)"

            # 3. 조도 및 JIG 상태 종합 알람 처리
            is_bright_ok = self.current_brightness >= self.min_brightness_limit

            if not is_bright_ok:
                self.jig_alarm_label.setText("JIG 조도 미달 경고!")
                self.jig_alarm_label.setStyleSheet(
                    "color: white; background-color: #DC2626; font-weight: bold; padding: 4px; border-radius: 6px;")
            elif not is_yolo_ok:
                self.jig_alarm_label.setText(yolo_msg)
                self.jig_alarm_label.setStyleSheet(
                    "color: #991B1B; background-color: #FEE2E2; font-weight: bold; padding: 4px; border-radius: 6px;")
            else:
                self.jig_alarm_label.setText(f"JIG 상태 정상 - {yolo_msg}")
                self.jig_alarm_label.setStyleSheet(
                    "color: #0369A1; background-color: #E0F2FE; font-weight: bold; padding: 4px; border-radius: 6px;")

            # 4. PyQt 프리뷰 화면 출력
            rgb_image = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb_image.data, cw, ch, cw * 3, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_img))

        except Exception as e:
            print(f"프레임 업데이트 예외: {e}")

    # =========================================================================
    # [수정] 모델 관리: 캡처 영역은 건드리지 않고 오직 "임계값"만 변경/저장
    # =========================================================================
    def load_selected_model(self):
        try:
            idx = self.combo_models.currentIndex() + 1
            slot_key = f"model_{idx}"
            cfg = self.models_data.get(slot_key, self.get_default_model_config())

            self.is_loading_config = True
            self.block_ui_signals(True)

            self.spin_thresh_r.setValue(cfg.get("target_score_r", 1000.0))
            self.spin_thresh_c.setValue(cfg.get("target_score_c", 1000.0))
            self.spin_thresh_l.setValue(cfg.get("target_score_l", 1000.0))
            self.spin_thresh_dev.setValue(cfg.get("max_allowed_dev_rl", 20.0))

            self.block_ui_signals(False)
            self.update_runtime_vars_from_ui()
            self.is_loading_config = False

            self.lbl_active_model.setText(f"현재 선택된 모델: Model {idx}")
            self.status_label.setText(f"Model {idx} 임계값 불러오기 성공")
        except Exception as e:
            print(f"모델 불러오기 오류: {e}")
            self.status_label.setText("모델 불러오기 실패")

    def save_selected_model(self):
        try:
            idx = self.combo_models.currentIndex() + 1
            slot_key = f"model_{idx}"

            self.models_data[slot_key] = {
                "target_score_r": self.spin_thresh_r.value(),
                "target_score_c": self.spin_thresh_c.value(),
                "target_score_l": self.spin_thresh_l.value(),
                "max_allowed_dev_rl": self.spin_thresh_dev.value()
            }
            self.save_configs_to_file()

            self.lbl_active_model.setText(f"현재 선택된 모델: Model {idx}")
            self.status_label.setText(f"Model {idx} 임계값 저장 완료!")
        except Exception as e:
            print(f"모델 저장 오류: {e}")

    def save_to_excel(self, result_text, mean_r, mean_c, mean_l, dev_rl, region_scores):
        try:
            target_dir = self.get_save_dir()
            excel_path = os.path.join(target_dir, "Focus_Test_Results.xlsx")
            new_data = {
                "검사 시간": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "판정": [result_text.split()[0]],
                "Right 평균(%)": [round(mean_r, 1)],
                "Center 평균(%)": [round(mean_c, 1)],
                "Left 평균(%)": [round(mean_l, 1)],
                "R-L 편차율(%)": [round(dev_rl, 1)],
                "측정 조도(LUX)": [round(self.current_brightness, 1)]
            }
            idx = 0
            for r in range(1, 4):
                for c in range(1, 4):
                    new_data[f"R{r}{c}(%)"] = [round(region_scores[idx], 1)]
                    idx += 1
            df_new = pd.DataFrame(new_data)
            if os.path.exists(excel_path):
                df_existing = pd.read_excel(excel_path)
                df_updated = pd.concat([df_existing, df_new], ignore_index=True)
                df_updated.to_excel(excel_path, index=False)
            else:
                df_new.to_excel(excel_path, index=False)
        except Exception as e:
            print(f"엑셀 저장 오류: {e}")

    # =========================================================================
    # [수정] 가장 기본적이고 안정적인 3x3 표준 검사 (YOLO 크롭 로직 제거)
    # =========================================================================
    def run_focus_test(self):
        try:
            self.update_runtime_vars_from_ui()  # 검사 직전 UI 수치 최신화 반영

            if self.current_frame is None:
                self.status_label.setText("오류: 영상 없음")
                return

            working_frame = self.current_frame.copy()
            gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
            save_img = working_frame.copy()
            cv2.rectangle(save_img, (0, 0), (self.target_w - 1, self.target_h - 1), (0, 255, 0), 4)

            x_lines_save, y_lines_save = self.get_grid_lines_100(self.target_w, self.target_h)
            region_raw_scores = []

            for r in range(3):
                for c in range(3):
                    x1, x2 = x_lines_save[c], x_lines_save[c + 1]
                    y1, y2 = y_lines_save[r], y_lines_save[r + 1]

                    cell_gray = gray[y1:y2, x1:x2]
                    raw_score = self.calculate_focus_score(cell_gray)
                    region_raw_scores.append(raw_score)

                    self.region_labels[r][c].setText(f"R{r + 1}{c + 1}\nRaw: {raw_score:.1f}")
                    cv2.rectangle(save_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(save_img, f"Raw:{raw_score:.1f}", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

            raw_mean_r = float(np.mean([region_raw_scores[0], region_raw_scores[3], region_raw_scores[6]]))
            raw_mean_c = float(np.mean([region_raw_scores[1], region_raw_scores[4], region_raw_scores[7]]))
            raw_mean_l = float(np.mean([region_raw_scores[2], region_raw_scores[5], region_raw_scores[8]]))

            pct_r = (raw_mean_r / self.target_score_r * 100.0) if self.target_score_r > 0 else 0.0
            pct_c = (raw_mean_c / self.target_score_c * 100.0) if self.target_score_c > 0 else 0.0
            pct_l = (raw_mean_l / self.target_score_l * 100.0) if self.target_score_l > 0 else 0.0

            raw_rl_avg = np.mean([raw_mean_r, raw_mean_l])
            dev_rl_percent = (abs(raw_mean_r - raw_mean_l) / raw_rl_avg * 100.0) if raw_rl_avg > 0 else 0.0

            is_r_pass = pct_r >= 100.0
            is_c_pass = pct_c >= 100.0
            is_l_pass = pct_l >= 100.0
            is_dev_pass = dev_rl_percent <= self.max_allowed_dev_rl
            is_bright_pass = self.current_brightness >= self.min_brightness_limit

            if is_r_pass and is_c_pass and is_l_pass and is_dev_pass and is_bright_pass:
                result_text = "PASS"
                status_msg = "검사 성공 (엑셀 저장 완료)"
                overlay_color = (0, 255, 0)
            else:
                result_text = "FAIL (NG)"
                fail_reasons = []
                if not is_r_pass: fail_reasons.append("Right 미달")
                if not is_c_pass: fail_reasons.append("Center 미달")
                if not is_l_pass: fail_reasons.append("Left 미달")
                if not is_dev_pass: fail_reasons.append("R-L 편차 초과")
                if not is_bright_pass: fail_reasons.append("JIG 조도 부족")
                status_msg = f"불합격: {', '.join(fail_reasons)}"
                overlay_color = (0, 0, 255)

            self.result_header_label.setText(result_text)
            self.total_score_label.setText(
                f"R: {raw_mean_r:.1f} ({pct_r:.1f}%) | C: {raw_mean_c:.1f} ({pct_c:.1f}%) | L: {raw_mean_l:.1f} ({pct_l:.1f}%)")
            self.dev_label.setText(f"R-L 편차율: {dev_rl_percent:.1f}% (허용: {self.max_allowed_dev_rl:.1f}%)")
            self.status_label.setText(status_msg)
            self.update_ui_fonts()

            cv2.putText(save_img,
                        f"[{result_text}] R:{pct_r:.1f}% C:{pct_c:.1f}% L:{pct_l:.1f}% / DEV:{dev_rl_percent:.1f}%",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)

            target_dir = self.get_save_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"FocusResult_{result_text.split()[0]}_{pct_r:.0f}pct_{timestamp}.jpg"

            is_success, encoded_img = cv2.imencode(".jpg", save_img)
            if is_success:
                with open(os.path.join(target_dir, file_name), mode='wb') as f:
                    encoded_img.tofile(f)

            self.save_to_excel(result_text, pct_r, pct_c, pct_l, dev_rl_percent, region_raw_scores)

        except Exception as e:
            print(f"검사 도중 오류 발생:\n{traceback.format_exc()}")
            self.status_label.setText("검사 중 오류 발생")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AutoFocusTestApp()
    ex.show()
    sys.exit(app.exec_())