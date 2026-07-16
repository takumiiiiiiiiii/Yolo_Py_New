# ******************************************************************************
#  pyorbbecsdk 初級サンプル 03 — カラーとデプスのアライメント（マルチスレッド版）
#
#  【マルチスレッド化のポイント】
#    - メインスレッド : パイプラインからのフレーム取得・アライメント・デプス変換のみを担当
#                        （ここは高頻度・低遅延で回したいのでYOLO推論を絶対に挟まない）
#    - 推論スレッド    : YOLOによる姿勢推定 → 目の3D座標計算 → TCP送信 を担当
#                        （YOLOは重いので、メインループをブロックしないよう別スレッドへ）
#    - 受け渡し        : queue.Queue(maxsize=1) を使用。
#                        推論が追いつかない場合は「古いフレームを捨てて最新だけ入れる」
#                        方式にすることで、キューが詰まってメイン側が待たされることがない
#                        （＝常に最新のフレームだけを処理する「フレームスキップ」構成）
#
#  依存パッケージ: numpy, opencv-python, utils.py, ultralytics
#
#  実行方法:
#    python 03_color_and_depth_aligned_mt.py
#    python 03_color_and_depth_aligned_mt.py --hw
#    python 03_color_and_depth_aligned_mt.py --no-tcp
# ******************************************************************************

import math
import argparse
import os
import sys
import time
import threading
import queue
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import mediapipe as mp
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

# MediaPipe Face Mesh（現状未使用だが元コードを維持）
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

HOST = "127.0.0.1"
PORT = 50000

client = None  # main() 内で必要な場合のみ接続する

# --- 設定用の定数 ---
ESC_KEY = 27
MIN_DEPTH = 20
MAX_DEPTH = 10000

# ---------------------------------------------------------------------------
# デバイス選択（M1/M2 Mac の GPU = MPS を優先的に使用）
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    YOLO_DEVICE = "mps"
elif torch.cuda.is_available():
    YOLO_DEVICE = "cuda"
else:
    YOLO_DEVICE = "cpu"
print(f"YOLO推論デバイス: {YOLO_DEVICE}")

# YOLOモデルの読み込み
YoloModel = YOLO("yolo11n-pose.pt")
# モデルを先にデバイスへ乗せておく（毎推論ごとの転送コストを避ける）
YoloModel.to(YOLO_DEVICE)

# ---------------------------------------------------------------------------
# マルチスレッド用の共有リソース
# ---------------------------------------------------------------------------
frame_queue = queue.Queue(maxsize=1)   # (color_image, depth_data) を1つだけ保持
display_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()
tcp_lock = threading.Lock()

# カメラ内部パラメータ（main() 内で設定してからスレッド開始する）
_intrinsics = {"fx": None, "fy": None, "cx": None, "cy": None}


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def euclidean_distance(p1, p2):
    """2つの3D座標(cm)間のユークリッド距離(cm)を計算"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def pixel_to_world(u, v, depth_mm, fx, fy, cx, cy):
    """
    u, v      : ピクセル座標（反転していない元画像上）
    depth_mm  : そのピクセルの深度値（mm）
    戻り値    : (X, Y, Z) メートル単位
    """
    Z = depth_mm / 10.0  # mm → m
    if Z <= 0:
        return None
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z


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


# ---------------------------------------------------------------------------
# 推論スレッド本体（YOLO推論 + 座標計算 + TCP送信）
# ---------------------------------------------------------------------------

def yolo_worker(use_tcp: bool):
    """
    frame_queue から最新フレームを取り出し、YOLO推論・目の3D座標算出・TCP送信を行う。
    キューが空なら短時間待って再チェックするだけなので、CPUを無駄に使わない。
    メインループとは完全に非同期で動く。
    """
    global client

    while not stop_event.is_set():
        try:
            color_image, depth_data = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        fx = _intrinsics["fx"]
        fy = _intrinsics["fy"]
        cx = _intrinsics["cx"]
        cy = _intrinsics["cy"]
        if fx is None:
            # --hw モードなど、内部パラメータ未取得の場合は座標計算をスキップ
            continue

        try:
            results = YoloModel(color_image, imgsz=640, device="mps", verbose=False)
            results = results[0]
        except Exception as e:
            print(f"YOLO推論エラー: {e}")
            continue

        if results.keypoints is None:
            continue
        kpts = results.keypoints.xy.cpu().numpy()
        if len(kpts) == 0:
            continue
        # cv2.imshow("YOLO Keypoints", results.plot())
        left_eye = kpts[0][1]
        right_eye = kpts[0][2]

        lx, ly = int(left_eye[0]), int(left_eye[1])
        rx, ry = int(right_eye[0]), int(right_eye[1])

        h, w = depth_data.shape[:2]
        if not (0 <= ly < h and 0 <= lx < w and 0 <= ry < h and 0 <= rx < w):
            continue
        # --- ここで描画フレームをdisplay_queueへ ---
        annotated = results.plot()
        if display_queue.full():
            try:
                display_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            display_queue.put_nowait(annotated)
        except queue.Full:
            pass
        left_depth = depth_data[ly, lx]
        right_depth = depth_data[ry, rx]

        left_world = pixel_to_world(lx, ly, left_depth, fx, fy, cx, cy)
        right_world = pixel_to_world(rx, ry, right_depth, fx, fy, cx, cy)
        print(f"Left Eye 3D: {left_world}, Right Eye 3D: {right_world}")
        if left_world is None or right_world is None:
            continue

        eye_center = (
            (left_world[0] + right_world[0]) / 2,
            (left_world[1] + right_world[1]) / 2,
            (left_world[2] + right_world[2]) / 2,
        )
        XL, YL, ZL = left_world
        XR, YR, ZR = right_world
        message = f"{XL:.3f},{YL:.3f},{ZL:.3f},{XR:.3f},{YR:.3f},{ZR:.3f}\n"

        if use_tcp and client is not None:
            try:
                with tcp_lock:
                    client.sendall(message.encode())
            except OSError as e:
                print(f"TCP送信エラー: {e}")


def push_frame_for_inference(color_image, depth_data):
    """
    推論スレッドへフレームを渡す。キューが満杯（＝推論が追いついていない）場合は
    古いフレームを捨てて最新のものに差し替える。これによりメインループは
    絶対にブロックされない。
    """
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    try:
        frame_queue.put_nowait((color_image, depth_data))
    except queue.Full:
        pass


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------

def main():
    frame_count = 0
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("Device Not Found! Please connect an Orbbec camera and try again.")
        return

    parser = argparse.ArgumentParser(description="Color + Depth aligned viewer (multithreaded)")
    parser.add_argument("--hw", action="store_true", help="Use hardware D2C alignment")
    parser.add_argument("--no-tcp", action="store_true", help="Do not connect to the TCP server")
    args = parser.parse_args()

    use_tcp = not args.no_tcp

    global client
    if use_tcp:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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

    window_name = "YoloKeypoints"
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
    # 推論スレッド開始（YOLO推論 + TCP送信はここで非同期に走る）
    # ------------------------------------------------------------------
    inference_thread = threading.Thread(target=yolo_worker, args=(use_tcp,), daemon=True)
    inference_thread.start()

    # ------------------------------------------------------------------
    # フレーム取得ループ（メインスレッド：ここは軽い処理のみ）
    # ------------------------------------------------------------------
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(1000)
                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                if not args.hw:
                    frames = align_filter.process(frames)
                    if not frames:
                        continue
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    continue

                try:
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                        (depth_frame.get_height(), depth_frame.get_width())
                    )
                except ValueError:
                    continue

                depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()

                if args.hw:
                    depth_data = np.clip(depth_data, MIN_DEPTH, MAX_DEPTH)
                else:
                    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)

                # -- ステータス表示用（任意） --
                depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
                h, w = color_image.shape[:2]
                if depth_vis.shape[:2] != (h, w):
                    depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)

                blend_alpha = alpha if args.hw else 0.5
                overlay = cv2.addWeighted(color_image, 1 - blend_alpha, depth_vis, blend_alpha, 0)

                if args.hw:
                    status = f"HW D2C: {'ON' if enable_hw_d2c else 'OFF'}  alpha={blend_alpha:.1f}"
                else:
                    mode_str = "D2C (Depth To Color)" if align_mode == 0 else "C2D (Color To Depth)"
                    sync_str = "Sync: ON" if enable_sync else "Sync: OFF"
                    status = f"{mode_str} | {sync_str}"

                cv2.putText(color_image, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # -- YOLO推論は毎フレームではなく間引いて推論スレッドへ渡す --
                frame_count += 1
                if frame_count % 2 == 0:
                    # copy() で推論スレッドとメインスレッドがデータを共有しないようにする
                    push_frame_for_inference(color_image.copy(), depth_data.copy())
                
                # -- YOLOランドマーク表示（推論スレッドから受け取り） --
                try:
                    annotated_frame = display_queue.get_nowait()
                    cv2.imshow(window_name, annotated_frame)
                except queue.Empty:
                    pass


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
        # ------------------------------------------------------------------
        # 終了処理：推論スレッドを止めてからパイプライン・ウィンドウを閉じる
        # ------------------------------------------------------------------
        stop_event.set()
        inference_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        pipeline.stop()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()