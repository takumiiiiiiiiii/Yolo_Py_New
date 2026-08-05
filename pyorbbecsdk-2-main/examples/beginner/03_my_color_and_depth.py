# ******************************************************************************
#  pyorbbecsdk 初級サンプル 03 — カラーとデプスのアライメント（マルチスレッド版・深度のみ）
#
#  YOLO(ultralytics)関連の処理をすべて削除し、深度画像だけで頭頂点を推定する
#  estimate_head_vertex_NoYolo / Un_yolo_worker 経路のみを残した版。
#
#  依存パッケージ: numpy, opencv-python, utils.py
#
#  実行方法:
#    python 03_my_color_and_depth_noyolo.py
#    python 03_my_color_and_depth_noyolo.py --hw
#    python 03_my_color_and_depth_noyolo.py --no-tcp
# ******************************************************************************


import time
import math
import argparse
import os
import sys
import threading
import queue
from collections import deque, namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import open3d as o3d

import socket
from utils import frame_to_bgr_image


from pyorbbecsdk import OBAlignMode  # type: ignore
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

# --- 設定用の定数 ---
ESC_KEY = 27

HOST = "127.0.0.1"
PORT = 50000

# Yoloなしの場合の深度閾値(頭部として扱う距離レンジ, mm)
MAX_DEPTH_NOYOLO = 1500
MIN_DEPTH_NOYOLO = 400

client = None  # main() 内で必要な場合のみ接続する

HEAD_TOP_TO_EYE_DROP_CM = 12.0   # 頭頂から目までの推定オフセット(決め打ち)
HALF_IPD_CM = 3.2                # 目の左右間隔の半分(決め打ち)


HeadVertexResult = namedtuple(
    "HeadVertexResult", ["world", "uv", "box_region"]
)


def fixed_eye_world_positions_no_yaw(head_world):
    """向きを無視して、頭頂点からの定数オフセットだけで目の位置を出す"""
    Xh, Yh, Zh = head_world
    left = (Xh - HALF_IPD_CM, Yh + HEAD_TOP_TO_EYE_DROP_CM, Zh)
    right = (Xh + HALF_IPD_CM, Yh + HEAD_TOP_TO_EYE_DROP_CM, Zh)
    return left, right


def world_to_pixel(X, Y, Z, fx, fy, cx, cy):
    """3D座標をピクセル座標に変換する関数"""
    if Z <= 0:
        return None
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return int(u), int(v)


def pixel_to_world(u, v, depth_mm, fx, fy, cx, cy):
    """
    u, v      : ピクセル座標（反転していない元画像上）
    depth_mm  : そのピクセルの深度値（mm）
    戻り値    : (X, Y, Z) cm 単位（depth_mm / 10.0 で mm→cm 変換）
    """
    Z = depth_mm / 10.0  # mm → cm
    if Z <= 0:
        return None
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z


def estimate_head_vertex_NoYolo(depth_raw, depth_scale, fx, fy, cx, cy):
    """深度画像だけから頭の位置を推定する関数(YOLO不使用、NumPyベクトル化版)"""
    region = depth_raw.astype(np.float32) * depth_scale

    # 有効範囲(MIN_DEPTH_NOYOLO 〜 MAX_DEPTH_NOYOLO)内のピクセルを一括判定
    valid = (region > MIN_DEPTH_NOYOLO) & (region < MAX_DEPTH_NOYOLO)

    ys, xs = np.nonzero(valid)
    if len(ys) == 0:
        return None

    # 画像上で最も上(v最小)の行を頭頂の行とみなす
    top_row = ys.min()
    row_xs = xs[ys == top_row]
    # その行内で有効なピクセルの中心を頭頂のx座標とする
    top_col = int(np.mean(row_xs))
    u, v = top_col, int(top_row)
    # 深度値はその行で実際に有効だったピクセルから取る(中心座標そのものが
    # 有効とは限らないため、row_xs[0] の実測値を使う)
    depth_mm = float(region[top_row, row_xs[0]])

    world = pixel_to_world(u, v, depth_mm, fx, fy, cx, cy)
    if world is None:
        return None

    return HeadVertexResult(
        world=world, uv=(u, v),
        box_region=(0, 0, region.shape[1], region.shape[0])
    )


# ---------------------------------------------------------------------------
# マルチスレッド用の共有リソース
# ---------------------------------------------------------------------------
frame_queue = queue.Queue(maxsize=1)   # (color_image, depth_raw, depth_scale) を1つだけ保持
display_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
tcp_lock = threading.Lock()

# カメラ内部パラメータ（main() 内で設定してからスレッド開始する）
_intrinsics = {"fx": None, "fy": None, "cx": None, "cy": None}

# 計測用バッファ
_main_timings = {
    "align": deque(maxlen=120),
    "color": deque(maxlen=120),
    "depth_copy": deque(maxlen=120),
    "push": deque(maxlen=120),
    "total": deque(maxlen=120),
}


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def push_frame_for_inference(color_image, depth_raw, depth_scale):
    """
    推論スレッドへフレームを渡す。キューが満杯（＝処理が追いついていない）場合は
    古いフレームを捨てて最新のものに差し替える。メインループは絶対にブロックされない。
    """
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        frame_queue.put_nowait((color_image, depth_raw, depth_scale))
    except queue.Full:
        pass


def euclidean_distance(p1, p2):
    """2つの3D座標(cm)間のユークリッド距離(cm)を計算"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def get_hw_stream_config(pipeline: Pipeline):
    config = Config()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        for i in range(len(profile_list)):
            color_profile = profile_list[i]
            if color_profile.get_format() != OBFormat.RGB:
                continue
            hw_depth_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
            if len(hw_depth_list) == 0:
                continue
            config.enable_stream(hw_depth_list[0])
            config.enable_stream(color_profile)
            config.set_align_mode(OBAlignMode.HW_MODE)

            # --hw モードでも内部パラメータを取得（座標計算を有効にする）
            try:
                intrinsic = color_profile.as_video_stream_profile().get_intrinsic()
                _intrinsics["fx"] = intrinsic.fx
                _intrinsics["fy"] = intrinsic.fy
                _intrinsics["cx"] = intrinsic.cx
                _intrinsics["cy"] = intrinsic.cy
            except Exception as e:
                print(f"[HW] 内部パラメータ取得失敗: {e}")

            return config
    except Exception as e:
        print(f"HW D2C config error: {e}")
    return None


def switch_hw_d2c(pipeline: Pipeline, config: Config, enable: bool):
    pipeline.stop()
    time.sleep(0.1)
    config.set_align_mode(OBAlignMode.HW_MODE if enable else OBAlignMode.DISABLE)
    print(f"Hardware D2C: {'Enabled' if enable else 'Disabled'}")
    pipeline.start(config)


def _log_timings():
    """直近の計測結果の平均を表示（デバッグ用）"""
    def avg(d):
        return sum(d) / len(d) * 1000 if d else 0.0  # s → ms
    print(
        f"[TIMING ms] "
        f"main(total={avg(_main_timings['total']):.1f} "
        f"align={avg(_main_timings['align']):.1f} "
        f"color={avg(_main_timings['color']):.1f} "
        f"depth_copy={avg(_main_timings['depth_copy']):.1f} "
        f"push={avg(_main_timings['push']):.1f})"
    )


# ---------------------------------------------------------------------------
# 推論スレッド本体（深度のみで頭部推定 + TCP送信）
# ---------------------------------------------------------------------------

def Un_yolo_worker(use_tcp: bool):
    """
    frame_queue から最新フレームを取り出し、深度のみで頭頂点・目の3D座標算出・TCP送信を行う。
    """
    global client
    report_ts = time.perf_counter()

    while not stop_event.is_set():
        try:
            color_image, depth_raw, depth_scale = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        fx = _intrinsics["fx"]
        fy = _intrinsics["fy"]
        cx = _intrinsics["cx"]
        cy = _intrinsics["cy"]
        if fx is None:
            continue

        result = estimate_head_vertex_NoYolo(depth_raw, depth_scale, fx, fy, cx, cy)
        if result is None:
            continue
        head_world, head_uv, (rx1, ry1, rx2, r_head_bottom) = result
        left_world, right_world = fixed_eye_world_positions_no_yaw(head_world)

        # 描画フレーム（TCP でなければ表示用に生成）
        if use_tcp is False:
            annotated = color_image.copy()
            # 探索対象領域(全画面)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, r_head_bottom), (0, 200, 0), 1)

            # 頭頂点
            cv2.circle(annotated, head_uv, 6, (0, 255, 255), -1)
            cv2.putText(annotated, f"Z={head_world[2]:.1f}cm", (head_uv[0] + 8, head_uv[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 推定した目の位置(3D→2D逆投影)
            for label, w_pos in (("L", left_world), ("R", right_world)):
                px = world_to_pixel(*w_pos, fx, fy, cx, cy)
                if px is not None:
                    cv2.circle(annotated, px, 5, (255, 0, 255), -1)
                    cv2.putText(annotated, label, (px[0] + 6, px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                display_queue.put_nowait(annotated)
            except queue.Full:
                pass

        XL, YL, ZL = left_world
        XR, YR, ZR = right_world
        message = f"{XL:.3f},{YL:.3f},{ZL:.3f},{XR:.3f},{YR:.3f},{ZR:.3f}\n"
        if use_tcp and client is not None:
            try:
                with tcp_lock:
                    client.sendall(message.encode())
            except OSError as e:
                print(f"TCP送信エラー: {e}")

        # 2秒おきにタイミングログ
        now = time.perf_counter()
        if now - report_ts > 2.0:
            _log_timings()
            report_ts = now


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def main():
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("Device Not Found! Please connect an Orbbec camera and try again.")
        return

    parser = argparse.ArgumentParser(description="Color + Depth aligned viewer (multithreaded, no YOLO)")
    parser.add_argument("--hw", action="store_true", help="Use hardware D2C alignment")
    parser.add_argument("--no-tcp", action="store_true", help="Do not connect to the TCP server")
    args = parser.parse_args()

    use_tcp = not args.no_tcp
    show_window = not use_tcp  # TCP 送信のみなら GUI をスキップして高速化

    global client
    if use_tcp:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("接続中...")
        try:
            client.connect((HOST, PORT))
            print("接続成功")
        except (ConnectionRefusedError, OSError) as e:
            print(f"TCP接続に失敗しました: {e}")
            print("--no-tcp オプションを付けて再実行するか、C++サーバーを先に起動してください。")
            return
    else:
        print("TCPなしモードで起動します（送信は行いません）")

    window_name = "HeadTracking"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    pipeline = Pipeline()
    config = None

    if args.hw:
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(f"Frame sync warning: {e}")

        config = get_hw_stream_config(pipeline)
        if config is None:
            print("ERROR: Hardware D2C is not supported on this device. Try without --hw.")
            return

        enable_hw_d2c = True
        alpha = 0.5
        alpha_step = 0.1

        print("\n========== Hardware D2C Align ==========")
        print("T       : Enable / Disable HW D2C")
        print("+ / -   : Adjust depth overlay transparency")
        print("Q / ESC : Quit")
        print("========================================\n")

    else:
        config = Config()
        align_mode = 0
        enable_sync = False

        try:
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            for i in range(len(profile_list)):
                p = profile_list[i].as_video_stream_profile()
                print(p.get_width(), p.get_height(), p.get_fps(), p.get_format())
            color_profile = profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
            config.enable_stream(color_profile)

            intrinsic = color_profile.get_intrinsic()
            _intrinsics["fx"] = intrinsic.fx
            _intrinsics["fy"] = intrinsic.fy
            _intrinsics["cx"] = intrinsic.cx
            _intrinsics["cy"] = intrinsic.cy
            print(f"fx={intrinsic.fx}, fy={intrinsic.fy}, cx={intrinsic.cx}, cy={intrinsic.cy}")

            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)

            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception as e:
            print(f"Stream configuration error: {e}")
            return

        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        if show_window:
            print("\nControls:")
            print("  T       — Toggle align direction  (D2C ↔ C2D)")
            print("  F       — Toggle frame sync        (ON / OFF)")
            print("  Q / ESC — Quit\n")

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Pipeline start error: {e}")
        return

    # ------------------------------------------------------------------
    # 推論スレッド開始（深度のみでの頭部推定 + TCP送信はここで非同期に走る）
    # ------------------------------------------------------------------
    inference_thread = threading.Thread(target=Un_yolo_worker, args=(use_tcp,), daemon=True)
    inference_thread.start()

    report_ts = time.perf_counter()

    # ------------------------------------------------------------------
    # フレーム取得ループ（メインスレッド：ここは軽い処理のみ）
    # ------------------------------------------------------------------
    try:
        while True:
            t_loop = time.perf_counter()
            try:
                frames = pipeline.wait_for_frames(1000)
                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                if not args.hw:
                    t_align = time.perf_counter()
                    frames = align_filter.process(frames)
                    _main_timings["align"].append(time.perf_counter() - t_align)
                    if not frames:
                        continue
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                t_color = time.perf_counter()
                color_image = frame_to_bgr_image(color_frame)
                _main_timings["color"].append(time.perf_counter() - t_color)
                if color_image is None:
                    continue

                # 深度：uint16 コピーだけ取得（スケール適用は推論スレッド側で行う）
                t_depth = time.perf_counter()
                try:
                    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                        depth_frame.get_height(), depth_frame.get_width()
                    ).copy()  # 内部バッファ再利用対策で必ず copy
                except ValueError:
                    continue
                depth_scale = depth_frame.get_depth_scale()
                _main_timings["depth_copy"].append(time.perf_counter() - t_depth)

                # ステータス描画は表示時のみ
                if show_window:
                    if args.hw:
                        status = f"HW D2C: {'ON' if enable_hw_d2c else 'OFF'}  alpha={alpha:.1f}"
                    else:
                        mode_str = "D2C (Depth To Color)" if align_mode == 0 else "C2D (Color To Depth)"
                        sync_str = "Sync: ON" if enable_sync else "Sync: OFF"
                        status = f"{mode_str} | {sync_str}"
                    cv2.putText(color_image, status, (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 推論スレッドへ最新フレームを渡す
                t_push = time.perf_counter()
                push_frame_for_inference(color_image, depth_raw, depth_scale)
                _main_timings["push"].append(time.perf_counter() - t_push)

                # 推定結果の表示（推論スレッドから受け取り）
                if show_window:
                    try:
                        annotated_frame = display_queue.get_nowait()
                        cv2.imshow(window_name, annotated_frame)
                    except queue.Empty:
                        pass

                _main_timings["total"].append(time.perf_counter() - t_loop)

                # 2秒おきにタイミングログ
                now = time.perf_counter()
                if now - report_ts > 2.0:
                    _log_timings()
                    report_ts = now

                # GUI ポーリング（表示時のみ。TCP モードでは SIGINT で終了）
                if show_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ESC_KEY):
                        break
                    if args.hw:
                        if key in (ord("t"), ord("T")):
                            enable_hw_d2c = not enable_hw_d2c
                            switch_hw_d2c(pipeline, config, enable_hw_d2c)
                        elif key in (ord("+"), ord("=")):
                            alpha = min(1.0, alpha + alpha_step)
                            print(f"Alpha: {alpha:.2f}")
                        elif key in (ord("-"), ord("_")):
                            alpha = max(0.0, alpha - alpha_step)
                            print(f"Alpha: {alpha:.2f}")
                    else:
                        if key in (ord("t"), ord("T")):
                            align_mode = (align_mode + 1) % 2
                            if align_mode == 0:
                                align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                                print("Mode: Depth To Color")
                            else:
                                align_filter = AlignFilter(align_to_stream=OBStreamType.DEPTH_STREAM)
                                print("Mode: Color To Depth")
                        elif key in (ord("f"), ord("F")):
                            enable_sync = not enable_sync
                            if enable_sync:
                                pipeline.enable_frame_sync()
                                print("Frame sync: ON")
                            else:
                                pipeline.disable_frame_sync()
                                print("Frame sync: OFF")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Runtime error: {e}")
                continue
    finally:
        # 終了処理：推論スレッドを止めてからパイプライン・ウィンドウを閉じる
        stop_event.set()
        inference_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        pipeline.stop()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()